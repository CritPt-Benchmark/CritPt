import copy

import openai
from inspect_ai._util.content import ContentText
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool

from critpt.templates import ParsePrompt


async def solve_with_parse(generate, _current_state, __parse=True, _code_template=None):
    """
    _current_state --generate--> new_state ----------(preserve reasoning info)--------*--*
                                    | normalize (=remove reasoning info)              |  |
                                 new_state                                            |  |
                                    | (copy)                        (__parse=False)   |  v
                                    *--------------> parsing_state -------generate----[----> return (new_state, parsing_state)
                                                          |                           |
                                                          | (__parse=false)           v
                                                          *--------------------------------> return (new_state, parsing_state)
    :param _current_state:
    :param __parse: True: one-step; False: two-step + parsing step
    :return: the state AFTER parse. Note that _current_state will be modified in place for the step before parsing.
    """
    
    async def generate_and_normalize(_current_state, keep_reasoning_block=False):
        
        # (not golden) or multiturn_with_answer ~ multiturn ~ keep
        keep_reasoning_block = (not _current_state.metadata['config']['use_golden_for_prev_steps']) \
                                    or _current_state.metadata['multiturn_with_answer']

        # print("=== GENERATE_AND_NORMALIZE START ===")
        # print(f"Input state has {len(_current_state.messages)} messages")
        # print(f"Tools available: {(_current_state.tools, len(_current_state.tools)) if hasattr(_current_state, 'tools') and _current_state.tools else 0}")
        
        system_messages = 0
        for i, message in enumerate(_current_state.messages):
            # print(f"Message {i}: {type(message).__name__}")
            if isinstance(message, ChatMessageSystem):
                system_messages += 1
                # print(f"System message {system_messages}: {message.content}")
                # if system_messages == 2:
                #     break
        
        # print(f"System messages found: {system_messages}")
        
        # Check if we need to generate
        last_message = _current_state.messages[-1] if _current_state.messages else None
        needs_generation = (
            system_messages == 1 or 
            not last_message or 
            not isinstance(last_message, (ChatMessageAssistant, ChatMessageTool))
        )
        
        # print(f"Needs generation: {needs_generation}")
        # print(f"Last message type: {type(last_message).__name__ if last_message else 'None'}")
        # if type(last_message).__name__ == "ChatMessageAssistant":
        #     print(f"Last message content: {last_message.content if last_message else 'None'}")
        
        if needs_generation:
            # print("Calling generate()...")
            await generate(_current_state)
            # print(f"After generate: {len(_current_state.messages)} messages")
            # print(f"Last message type: {type(_current_state.messages[-1]).__name__}")
        else:
            print("Skip Generation. Current State is:")
            # print(_current_state)
        
        _current_state_backup = copy.deepcopy(_current_state)
        
        # DEBUG: Inspect the generated content
        last_msg = _current_state.messages[-1]
        # print(f"=== ANALYZING LAST MESSAGE ===")
        # print(f"Message type: {type(last_msg).__name__}")
        # print(f"Message role: {last_msg.role}")
        # print(f"Content type: {type(last_msg.content)}")
        # print(f"Last Message: {last_msg}")
        
        # if isinstance(last_msg.content, str):
        #     print(f"String content length: {len(last_msg.content)}")
        #     print(f"String content preview: {repr(last_msg.content[:200])}")
        # elif hasattr(last_msg.content, '__len__'):
        #     print(f"Structured content with {len(last_msg.content)} parts:")
        #     for i, part in enumerate(last_msg.content):
        #         print(f"  Part {i}: type={getattr(part, 'type', 'unknown')}, class={type(part).__name__}")
        #         if hasattr(part, 'text'):
        #             print(f"    Text length: {len(part.text)}")
        #             print(f"    Text preview: {repr(part.text[:200])}...{repr(part.text[-200:])}")
        #         elif hasattr(part, 'reasoning'):
        #             print(f"    Reasoning length: {len(part.reasoning)}")
        #             print(f"    Reasoning preview: {repr(part.reasoning[:200])}...{repr(part.reasoning[-200:])}")
        
        return_content = _current_state.messages[-1].content
        
        # Handle string content (unchanged)
        if isinstance(return_content, str):
            # print("String content - no normalization needed")
            return _current_state_backup, _current_state
        
        # Handle structured content
        if not hasattr(return_content, '__len__'):
            # print("WARNING: Unexpected content structure")
            return _current_state_backup, _current_state
            
        reasoning_block_indices = []
        last_text_block_index = -1

        for _content_idx in range(len(return_content)):
            
            content_part = return_content[_content_idx]

            content_type = getattr(content_part, 'type', 'unknown')
            # print(f"Processing part {_content_idx}: type={content_type}")
            
            if content_type == "reasoning":
                if keep_reasoning_block:
                    reasoning_block_indices.append(_content_idx)
            elif content_type == "text":
                last_text_block_index = _content_idx

        # print(f"Reasoning blocks: {reasoning_block_indices}")
        # print(f"Last text block: {last_text_block_index}")

        # check the order: remove any reasoning blocks after the last text block
        while not keep_reasoning_block and len(reasoning_block_indices) > 0 and reasoning_block_indices[-1] > last_text_block_index:
            reasoning_block_indices.pop(-1)

        new_content = []
        # if there is a valid reasoning block, then add that block
        if len(reasoning_block_indices) > 0:
            # legacy: only keep one
            # new_content.append(return_content[reasoning_block_indices[-1]])
            # keep all
            new_content.extend([return_content[reasoning_block_idx] 
                                            for reasoning_block_idx in reasoning_block_indices])
            # new_content.extend("\n\n".join([return_content[reasoning_block_idx] 
            #                                 for reasoning_block_idx in reasoning_block_indices]))

        # # if no output text is present, create an empty response
        # if last_text_block_index == -1:
        #     print("WARNING: No text block found, creating empty content")
        #     new_content.append(ContentText(text=""))
        # else:
        #     if not keep_reasoning_block:
        #         text_content = return_content[last_text_block_index].text
        #         print(f"Using text block with length: {len(text_content)}")
        #         new_content.append(ContentText(text=text_content))
        #     else:
        #         new_content.append(return_content[last_text_block_index])

        if not keep_reasoning_block:
            text_content = return_content[last_text_block_index].text
            # print(f"Using text block with length: {len(text_content)}")
            new_content.append(ContentText(text=text_content))
        else:
            new_content.append(return_content[last_text_block_index])

        _current_state.messages[-1] = ChatMessageAssistant(content=new_content)
        # print(f"=== NORMALIZE COMPLETE ===")
        return _current_state_backup, _current_state

    # print(f"=== SOLVE_WITH_PARSE START ===")
    # print(f"Parse mode: {__parse}")
    # print(f"Code template: {_code_template is not None}")
    
    generation_successful = True
    try:
        _current_state_backup, _current_state = await generate_and_normalize(_current_state)
    except openai.APITimeoutError:
        # print("TIMEOUT ERROR")
        _current_state_backup = copy.deepcopy(_current_state)
        _current_state.messages.append(ChatMessageSystem(content=ContentText(text="Assistant request timed out.")))
        generation_successful = False

    # copy the current state for parsing
    parsing_state = copy.deepcopy(_current_state)

    if not __parse and generation_successful:
        # print("=== STARTING PARSE STEP ===")
        _parsing_instruction = ChatMessageUser(
            content=ParsePrompt.default_system_prompt(code_template=_code_template)
        )
        parsing_state.messages.append(_parsing_instruction)
        
        # print("Calling generate() for parsing...")
        await generate(parsing_state)
        # print("Parse generation complete")
        
        # Check parsing result
        parse_result = parsing_state.messages[-1]
        # print(f"{parse_result=}")
        # print(f"Parse result type: {type(parse_result).__name__}")
        # if hasattr(parse_result, 'content'):
        #     if isinstance(parse_result.content, str):
        #         print(f"Parse result length: {len(parse_result.content)}")
        #     else:
        #         print(f"Parse result structure: {type(parse_result.content)}")

    # Since parsing_state is expected to be archived, we merge current_state_backup to preserve all reasoning info
    if not __parse and generation_successful:
        parsing_state.messages[-3] = _current_state_backup.messages[-1] # all previous turns are already sync'ed
    else:
        parsing_state.messages[-1] = _current_state_backup.messages[-1] # all previous turns are already sync'ed

    # print(f"=== SOLVE_WITH_PARSE RETURN ===")
    final_content = parsing_state.output.completion if hasattr(parsing_state, 'output') and parsing_state.output else "NO OUTPUT"
    # print(f"Final output length: {len(final_content) if final_content != 'NO OUTPUT' else 'NO OUTPUT'}")
    
    return _current_state_backup, parsing_state



