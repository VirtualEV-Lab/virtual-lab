"""Runs a meeting with LLM agents."""

import logging
import re
import time
from pathlib import Path
from typing import Literal, cast

from openai import BadRequestError, RateLimitError, OpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from tqdm import trange, tqdm

log = logging.getLogger(__name__)

from virtual_lab.agent import Agent
from virtual_lab.constants import CONSISTENT_TEMPERATURE, PUBMED_TOOL_DESCRIPTION
from virtual_lab.provider import chat_completion_kwargs, create_client, get_provider, resolve_model
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.utils import (
    count_discussion_tokens,
    count_tokens,
    get_summary,
    print_cost_and_time,
    run_tools,
    save_meeting,
)


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search: bool = False,
    return_summary: bool = False,
) -> str | None:
    """Runs a meeting with a LLM agents.

    :param meeting_type: The type of meeting.
    :param agenda: The agenda for the meeting.
    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param team_lead: The team lead for a team meeting (None for individual meeting).
    :param team_members: The team members for a team meeting (None for individual meeting).
    :param team_member: The team member for an individual meeting (None for team meeting).
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the meeting.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion.
    :param temperature: The sampling temperature.
    :param pubmed_search: Whether to include a PubMed search tool.
    :param return_summary: Whether to return the summary of the meeting.
    :return: The summary of the meeting (i.e., the last message) if return_summary is True, else None.
    """
    # Validate meeting type
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError("Individual meeting does not require team lead or team members")
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    # Start timing the meeting
    start_time = time.time()

    # Set up client (OpenAI by default, or DeepSeek when LLM_PROVIDER=deepseek)
    client = create_client()

    # Set up team
    if meeting_type == "team":
        assert team_lead is not None and team_members is not None
        team: list[Agent] = [team_lead] + list(team_members)
        primary_model = team_lead.model
        critic: Agent | None = None
    else:
        assert team_member is not None
        # Keep the critic on the same model as the team member. The package
        # SCIENTIFIC_CRITIC uses DEFAULT_MODEL (gpt-5.2), which OpenAI flags as
        # High-capability biology and rejects with error code `bio_policy`.
        critic = Agent(
            title=SCIENTIFIC_CRITIC.title,
            expertise=SCIENTIFIC_CRITIC.expertise,
            goal=SCIENTIFIC_CRITIC.goal,
            role=SCIENTIFIC_CRITIC.role,
            model=team_member.model,
        )
        team = [team_member, critic]
        primary_model = team_member.model

    requested_model = primary_model
    primary_model = resolve_model(primary_model)
    if primary_model != requested_model:
        print(
            f"LLM provider {get_provider()}: calling {primary_model} "
            f"(agents requested {requested_model})"
        )

    # Set up tools
    tools: list[ChatCompletionToolParam] | None = (
        [ChatCompletionToolParam(**PUBMED_TOOL_DESCRIPTION)] if pubmed_search else None  # type: ignore[misc]
    )

    # Set up tool token count
    tool_token_count = 0

    # Initialize discussion (list of agent/message dicts for output)
    discussion: list[dict[str, str]] = []

    # Initialize messages for API calls
    messages: list[ChatCompletionMessageParam] = []

    # Initial prompt for team meeting
    if meeting_type == "team":
        assert team_lead is not None and team_members is not None
        initial_content = team_meeting_start_prompt(
            team_lead=team_lead,
            team_members=team_members,
            agenda=agenda,
            agenda_questions=agenda_questions,
            agenda_rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            num_rounds=num_rounds,
        )
        messages.append({"role": "user", "content": initial_content})
        discussion.append({"agent": "User", "message": initial_content})

    # Loop through rounds
    for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
        round_num = round_index + 1

        # Loop through team and elicit responses
        for agent in tqdm(team, desc="Team"):
            # Prompt based on agent and round number
            if meeting_type == "team":
                assert team_lead is not None
                # Team meeting prompts
                if agent == team_lead:
                    if round_index == 0:
                        prompt = team_meeting_team_lead_initial_prompt(team_lead=team_lead)
                    elif round_index == num_rounds:
                        prompt = team_meeting_team_lead_final_prompt(
                            team_lead=team_lead,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                        )
                    else:
                        prompt = team_meeting_team_lead_intermediate_prompt(
                            team_lead=team_lead,
                            round_num=round_num - 1,
                            num_rounds=num_rounds,
                        )
                else:
                    prompt = team_meeting_team_member_prompt(
                        team_member=agent, round_num=round_num, num_rounds=num_rounds
                    )
            else:
                assert team_member is not None and critic is not None
                # Individual meeting prompts
                if agent == critic:
                    prompt = individual_meeting_critic_prompt(critic=critic, agent=team_member)
                else:
                    if round_index == 0:
                        prompt = individual_meeting_start_prompt(
                            team_member=team_member,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                            summaries=summaries,
                            contexts=contexts,
                        )
                    else:
                        prompt = individual_meeting_agent_prompt(critic=critic, agent=team_member)

            # Add prompt as user message
            messages.append({"role": "user", "content": prompt})
            discussion.append({"agent": "User", "message": prompt})

            # Build messages for this agent with their system prompt
            agent_messages: list[ChatCompletionMessageParam] = [agent.message] + messages

            # Call the chat completions API (with retry on rate limits)
            response = _call_with_retry(
                client, agent, agent_messages, temperature,
                tools=tools if tools else NOT_GIVEN,
            )

            # Get the response message
            response_message = response.choices[0].message

            # Check if the model wants to call tools
            if response_message.tool_calls:
                # Run the tools and get outputs
                tool_outputs, tool_messages = run_tools(tool_calls=response_message.tool_calls)

                # Update tool token count
                tool_token_count += sum(count_tokens(output) for output in tool_outputs)

                # Add the assistant's message with tool_calls to the messages.
                # DeepSeek thinking mode requires reasoning_content to be sent
                # back on later requests that include tools.
                messages.append(_assistant_api_message(response_message))

                # Add tool response messages
                for tool_msg in tool_messages:
                    messages.append(tool_msg)

                # Add tool outputs to discussion for visibility
                tool_output_content = "\n\n".join(tool_outputs)
                discussion.append({"agent": "Tool", "message": tool_output_content})

                # Make another API call with tool results
                agent_messages = [agent.message] + messages

                response = _call_with_retry(
                    client, agent, agent_messages, temperature,
                )
                response_message = response.choices[0].message

            # Extract the response content
            response_content = response_message.content or ""

            # Add response to messages and discussion
            messages.append(_assistant_api_message(response_message))
            discussion.append({"agent": agent.title, "message": response_content})

            # If final round, only team lead or team member responds
            if round_index == num_rounds:
                break

    # Count discussion tokens
    token_counts = count_discussion_tokens(discussion=discussion)

    # Add tool token count to total token count
    token_counts["tool"] = tool_token_count

    # Print cost and time
    # TODO: handle different models for different agents
    print_cost_and_time(
        token_counts=token_counts,
        model=primary_model,
        elapsed_time=time.time() - start_time,
    )

    # Save the discussion as JSON and Markdown
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )

    # Optionally, return summary
    if return_summary:
        return get_summary(discussion)

    return None


def _call_with_retry(
    client: OpenAI,
    agent: Agent,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    tools: list[ChatCompletionToolParam] | object = NOT_GIVEN,
    max_retries: int = 10,
    initial_wait: float = 10.0,
    max_wait: float = 120.0,
):
    """Call chat completions with exponential backoff on 429 RateLimitError.

    :param client: The OpenAI client.
    :param agent: The agent making the request (used for model and logging).
    :param messages: The messages to send.
    :param temperature: The sampling temperature.
    :param tools: Tool definitions, or NOT_GIVEN.
    :param max_retries: Maximum number of retries on rate-limit errors.
    :param initial_wait: Initial backoff wait in seconds.
    :param max_wait: Maximum backoff wait in seconds.
    :return: The chat completion response.
    """
    wait = initial_wait
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=resolve_model(agent.model),
                messages=messages,
                tools=tools,
                **chat_completion_kwargs(temperature),
            )
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            retry_after = _parse_retry_after(e)
            sleep_time = max(wait, retry_after) if retry_after else wait
            log.warning(
                "Rate limited (%s, %s), attempt %d/%d, sleeping %.1fs",
                agent.title, resolve_model(agent.model), attempt + 1, max_retries, sleep_time,
            )
            print(
                f"⏳ Rate limited ({agent.title}), "
                f"attempt {attempt + 1}/{max_retries}, "
                f"retrying in {sleep_time:.0f}s..."
            )
            time.sleep(sleep_time)
            wait = min(wait * 2, max_wait)
        except BadRequestError as e:
            _reraise_model_request_error(e, agent)


def _parse_retry_after(error: RateLimitError) -> float | None:
    """Extract the suggested wait time from OpenAI's 429 error message."""
    msg = str(error)
    match = re.search(r"try again in (\d+\.?\d*)s", msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return None


def _reraise_model_request_error(error: BadRequestError, agent: Agent) -> None:
    """Re-raise API 400s with context when OpenAI blocks a biology prompt."""
    error_body = error.body if isinstance(error.body, dict) else {}
    nested_error = error_body.get("error", {}) if isinstance(error_body, dict) else {}
    error_code = nested_error.get("code") if isinstance(nested_error, dict) else None
    if error_code == "bio_policy" or "bio_policy" in str(error):
        raise RuntimeError(
            f"OpenAI rejected this request as biological-risk content (bio_policy) "
            f"for model {agent.model} ({agent.title}). GPT-5.x models apply stricter "
            f"biology filters than GPT-4o. The published nanobody notebook uses "
            f"gpt-4o-2024-08-06; keep all agents on that model."
        ) from error
    raise error


def _assistant_api_message(response_message) -> ChatCompletionMessageParam:
    """Builds an assistant message to send back on the next turn.

    ``reasoning_content`` is included when the provider returns it. DeepSeek
    thinking mode rejects later tool-using requests that drop it.
    """
    message: dict[str, object] = {
        "role": "assistant",
        "content": response_message.content or "",
    }
    reasoning_content = getattr(response_message, "reasoning_content", None)
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if response_message.tool_calls:
        message["content"] = response_message.content
        message["tool_calls"] = [tc.model_dump() for tc in response_message.tool_calls]
    return cast(ChatCompletionMessageParam, message)
