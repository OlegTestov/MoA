import base64
from typing import List, Dict, Union
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
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        headers = {
            "content-type": "application/json",
            "x-api-key": self.config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        self.logger.debug("Request data:")
        self.logger.debug(json.dumps(data, indent=2))

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
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
        }

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        data = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        self.logger.debug("Request data:")
        self.logger.debug(json.dumps(data, indent=2))

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
        self, system_prompt: str, messages: List[Dict[str, str]]
    ) -> str:
        return self._call_anthropic_api(
            model="claude-3-5-sonnet-20240620",
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=1000,
            temperature=0.2,
        )

    def gpt_4o(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        return self._call_openai_api(
            model="gpt-4o",
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=1000,
            temperature=0.2,
        )

    def process_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"Error processing image: {e}")
            return f"Error: {e}"

    def run(self, user_prompt: Union[str, Dict]) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    user_prompt
                    if isinstance(user_prompt, str)
                    else json.dumps(user_prompt)
                ),
            }
        ]

        for layer_index, layer in enumerate(self.layers):
            self.logger.info(f"Processing layer {layer_index + 1}")

            if (
                layer_index < len(self.layers) - 1
            ):  # for all layers except the aggregation layer
                layer_responses = []
                for model_index, model in enumerate(layer, start=1):
                    response = model(self.prompts.moa_intermediate_system(), messages)
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
                messages.append({"role": "user", "content": self.prompts.moa_final_instruct(user_prompt)})

                final_response = layer[0](self.prompts.moa_final_system(), messages)
                return final_response

        self.logger.error("No valid response generated")
        return "Error: No valid response generated"


if __name__ == "__main__":
    moa = MoASystem()
    user_prompt = {
        "text": "At the event, there were 66 handshakes. If everyone shook hands with each other, how many people were at the event in total?",
        # "image": moa.process_image("path/to/your/image.jpg")
    }
    final_response = moa.run(user_prompt)
    print("Final response:")
    print(final_response)
