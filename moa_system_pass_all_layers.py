import base64
import copy
from typing import List, Dict
import json
import logging
import asyncio
import aiohttp

from constants import Prompts, Config


class MoASystem:
    def __init__(self):
        self.prompts = Prompts()
        self.config = Config()
        self.layers = [
            [self.claude_3_5_sonnet, self.gpt_4o],
            [self.claude_3_5_sonnet, self.gpt_4o],
            # [self.claude_3_5_sonnet, self.gpt_4o],
            [self.claude_3_5_sonnet],  # Aggregation layer
        ]

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.config.LOG_LEVEL)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.LOG_LEVEL)
        self.logger.addHandler(console_handler)

        if not self.config.ANTHROPIC_API_KEY or not self.config.OPENAI_API_KEY:
            self.logger.warning(
                "Please set ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables."
            )

    async def _call_anthropic_api_async(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        if user_text is None and user_image_path is None:
            raise ValueError("Either user_text or user_image_path must be provided")
        max_tokens = max_tokens or self.config.MAX_TOKENS
        temperature = temperature or self.config.TEMPERATURE

        headers = {
            "content-type": "application/json",
            "x-api-key": self.config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        }

        # Create the first message
        first_message = {"role": "user", "content": []}
        if user_text:
            first_message["content"].append({"type": "text", "text": user_text})
        if user_image_path:
            base64_image = self.process_image(user_image_path)
            first_message["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                }
            )

        anthropic_messages = [first_message] + messages

        data = {
            "model": model,
            "system": system_prompt,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        self.logger.debug("Request data:")
        logger_data = copy.deepcopy(data)
        if user_image_path:
            logger_data["messages"][0]["content"][1].pop("source")
        self.logger.debug(json.dumps(logger_data, indent=2))

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages", headers=headers, json=data
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()

                    self.logger.debug("Response data:")
                    self.logger.debug(json.dumps(response_data, indent=2))

                    if "content" in response_data and response_data["content"]:
                        return response_data["content"][0]["text"]
                    else:
                        self.logger.error(
                            f"Unexpected response structure: {response_data}"
                        )
                        return "Error: Unexpected response structure from Claude API"
        except aiohttp.ClientError as e:
            self.logger.error(f"Error in Claude API call: {e}")
            return f"Error: {e}"

    async def _call_openai_api_async(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        if user_text is None and user_image_path is None:
            raise ValueError("Either user_text or user_image_path must be provided")
        max_tokens = max_tokens or self.config.MAX_TOKENS
        temperature = temperature or self.config.TEMPERATURE
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
        }

        # Create the first message
        first_message = {"role": "user", "content": []}
        if user_text:
            first_message["content"].append({"type": "text", "text": user_text})
        if user_image_path:
            base64_image = self.process_image(user_image_path)
            first_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        openai_messages = [
            {"role": "system", "content": system_prompt},
            first_message,
        ] + messages

        data = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        self.logger.debug("Request data:")
        logger_data = copy.deepcopy(data)
        if user_image_path:
            logger_data["messages"][1]["content"][1].pop("image_url")
        self.logger.debug(json.dumps(logger_data, indent=2))

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()

                    self.logger.debug("Response data:")
                    self.logger.debug(json.dumps(response_data, indent=2))

                    if "choices" in response_data and response_data["choices"]:
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        self.logger.error(
                            f"Unexpected response structure: {response_data}"
                        )
                        return "Error: Unexpected response structure from OpenAI API"
        except aiohttp.ClientError as e:
            self.logger.error(f"Error in OpenAI API call: {e}")
            return f"Error: {e}"

    async def claude_3_5_sonnet(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        return await self._call_anthropic_api_async(
            model="claude-3-5-sonnet-20240620",
            system_prompt=system_prompt,
            messages=messages,
            user_text=user_text,
            user_image_path=user_image_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def gpt_4o(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        return await self._call_openai_api_async(
            model="gpt-4o",
            system_prompt=system_prompt,
            messages=messages,
            user_text=user_text,
            user_image_path=user_image_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def process_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"Error processing image: {e}")
            return f"Error: {e}"

    async def _process_layer_async(self, layer, messages, user_text, user_image_path):
        tasks = []
        for model_index, model in enumerate(layer, start=1):
            if model == self.claude_3_5_sonnet:
                async_model = self.claude_3_5_sonnet
            elif model == self.gpt_4o:
                async_model = self.gpt_4o
            else:
                raise ValueError(f"Unknown model: {model}")

            task = asyncio.create_task(
                async_model(
                    system_prompt=self.prompts.moa_intermediate_system(),
                    messages=messages,
                    user_text=user_text,
                    user_image_path=user_image_path,
                )
            )
            tasks.append((model_index, task))

        layer_responses = []
        for model_index, task in tasks:
            response = await task
            if not response.startswith("Error:"):
                layer_responses.append(f"Answer{model_index}: {response}")
            else:
                self.logger.warning(f"Skipping error response: {response}")

        return layer_responses

    def run(self, user_text: str = None, user_image_path: str = None) -> str:
        messages = []
        for layer_index, layer in enumerate(self.layers):
            self.logger.info(f"Processing layer {layer_index + 1}")

            if (
                layer_index < len(self.layers) - 1
            ):  # for all layers except the aggregation layer
                layer_responses = asyncio.run(
                    self._process_layer_async(
                        layer, messages, user_text, user_image_path
                    )
                )

                # Combine all responses from this layer into one message
                if layer_responses:
                    combined_response = "\n\n".join(layer_responses)
                    messages.append({"role": "assistant", "content": combined_response})

                if (
                    layer_index != len(self.layers) - 2
                ):  # for all layers except the last intermediate layer
                    messages.append(
                        {
                            "role": "user",
                            "content": self.prompts.moa_intermediate_instruct(),
                        }
                    )

            else:  # for the aggregation layer
                messages.append(
                    {
                        "role": "user",
                        "content": self.prompts.moa_final_instruct(user_text),
                    }
                )

                final_response = asyncio.run(
                    layer[0](
                        system_prompt=self.prompts.moa_final_system(),
                        messages=messages,
                        user_text=user_text,
                        user_image_path=user_image_path,
                    )
                )
                return final_response

        self.logger.error("No valid response generated")
        return "Error: No valid response generated"


if __name__ == "__main__":
    user_text = "Answer the question from the image"
    user_image_path = "image.jpg"

    moa = MoASystem()
    final_response = moa.run(user_text, user_image_path)
    print("Final response:")
    print(final_response)
