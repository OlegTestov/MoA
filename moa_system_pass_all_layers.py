import base64
import copy
from typing import List, Dict
import requests
import json
import logging

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

    def _call_anthropic_api(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
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
            response = requests.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            )
            response.raise_for_status()
            response_data = response.json()

            self.logger.debug("Response data:")
            self.logger.debug(json.dumps(response_data, indent=2))

            if "content" in response_data and response_data["content"]:
                return response_data["content"][0]["text"]
            else:
                self.logger.error(f"Unexpected response structure: {response_data}")
                return "Error: Unexpected response structure from Claude API"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in Claude API call: {e}")
            if hasattr(e, "response") and e.response is not None:
                self.logger.error(f"Response content: {e.response.text}")
            return f"Error: {e}"

    def _call_openai_api(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
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
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            )
            response.raise_for_status()
            response_data = response.json()

            self.logger.debug("Response data:")
            self.logger.debug(json.dumps(response_data, indent=2))

            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            else:
                self.logger.error(f"Unexpected response structure: {response_data}")
                return "Error: Unexpected response structure from OpenAI API"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in OpenAI API call: {e}")
            return f"Error: {e}"

    def claude_3_5_sonnet(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
    ) -> str:
        return self._call_anthropic_api(
            model="claude-3-5-sonnet-20240620",
            system_prompt=system_prompt,
            messages=messages,
            user_text=user_text,
            user_image_path=user_image_path,
            max_tokens=self.config.MAX_TOKENS,
            temperature=self.config.TEMPERATURE,
        )

    def gpt_4o(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_text: str = None,
        user_image_path: str = None,
    ) -> str:
        return self._call_openai_api(
            model="gpt-4o",
            system_prompt=system_prompt,
            messages=messages,
            user_text=user_text,
            user_image_path=user_image_path,
            max_tokens=self.config.MAX_TOKENS,
            temperature=self.config.TEMPERATURE,
        )

    def process_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"Error processing image: {e}")
            return f"Error: {e}"

    def run(self, user_text: str = None, user_image_path: str = None) -> str:
        messages = []
        for layer_index, layer in enumerate(self.layers):
            self.logger.info(f"Processing layer {layer_index + 1}")

            if (
                layer_index < len(self.layers) - 1
            ):  # for all layers except the aggregation layer
                layer_responses = []
                for model_index, model in enumerate(layer, start=1):
                    response = model(
                        system_prompt=self.prompts.moa_intermediate_system(),
                        messages=messages,
                        user_text=user_text,
                        user_image_path=user_image_path,
                    )
                    if not response.startswith("Error:"):
                        layer_responses.append(f"Answer{model_index}: {response}")
                    else:
                        self.logger.warning(
                            f"Skipping error response in layer {layer_index + 1}: {response}"
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

                final_response = layer[0](
                    system_prompt=self.prompts.moa_final_system(),
                    messages=messages,
                    user_text=user_text,
                    user_image_path=user_image_path,
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
