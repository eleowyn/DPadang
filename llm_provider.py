"""
LLM Provider using Hugging Face Inference API
Provides food-related information queries using Mistral model
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

class LLMProvider:
    """Hugging Face LLM provider for food information queries"""
    
    @staticmethod
    def ask_about_food(food_name: str, prompt: str) -> dict:
        """
        Get response about food from Hugging Face API
        
        Args:
            food_name: Name of the food (e.g., "Ayam Pop")
            prompt: User question (e.g., "Berikan resep lengkap")
        
        Returns:
            {
                "success": bool,
                "response": str,
                "provider": str (huggingface|error)
            }
        """
        
        # Use Hugging Face API
        result = LLMProvider.try_huggingface(food_name, prompt)
        if result["success"]:
            result["provider"] = "huggingface"
            return result
        
        # Failed
        return {
            "success": False,
            "response": "Maaf, LLM tidak tersedia saat ini. Silakan coba lagi nanti.",
            "provider": "error",
            "error": result.get("error")
        }
    
    @staticmethod
    def try_huggingface(food_name: str, prompt: str) -> dict:
        """
        Try Hugging Face Inference API with retry logic
        
        Setup:
        1. Go to https://huggingface.co/settings/tokens
        2. Create token with WRITE permissions (not Read!)
        3. Add to .env: HF_API_KEY=hf_xxxxx
        
        Free tier: 15,000 API calls/month
        """
        try:
            from huggingface_hub import InferenceClient
            
            hf_token = os.getenv("HF_API_KEY")
            if not hf_token:
                return {
                    "success": False,
                    "error": "HF_API_KEY tidak diset di .env. Ambil dari https://huggingface.co/settings/tokens"
                }
            
            print(f"🤖 Calling Hugging Face for: {food_name}")
            
            client = InferenceClient(api_key=hf_token, timeout=60)
            
            # Use conversational task instead of text_generation
            messages = [
                {
                    "role": "system",
                    "content": "Anda adalah ahli masakan Padang yang berpengalaman. Jawab pertanyaan dengan singkat tapi informatif (2-3 paragraf)."
                },
                {
                    "role": "user",
                    "content": f"Tentang makanan: {food_name}\nPertanyaan: {prompt}"
                }
            ]
            
            # Try multiple models in order of preference
            models_to_try = [
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",  # Best quality & speed
                "meta-llama/Llama-2-7b-chat-hf",                # Good quality
                "mistralai/Mistral-7B-Instruct-v0.2",           # Fast, reliable
                "HuggingFaceH4/zephyr-7b-beta"                  # Lightweight alternative
            ]
            
            last_error = None
            for model_name in models_to_try:
                try:
                    print(f"🔄 Trying model: {model_name.split('/')[-1]}")
                    message = client.chat_completion(
                        model=model_name,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7
                    )
                    response_text = message.choices[0].message.content
                    print(f"✅ HF response ({model_name.split('/')[-1]}): {len(response_text)} chars")
                    return {
                        "success": True,
                        "response": response_text.strip()
                    }
                except Exception as model_error:
                    last_error = model_error
                    print(f"⚠️  {model_name.split('/')[-1]} failed: {str(model_error)[:50]}")
                    continue
            
            # All models failed
            raise last_error if last_error else Exception("All models failed")
        
        except Exception as e:
            error_msg = str(e)
            # Check if it's a permission error
            if "403" in error_msg or "permission" in error_msg.lower():
                print(f"⚠️  Permission Error - Your token may need WRITE access")
                return {
                    "success": False,
                    "error": "Token permission denied. Create a new token with WRITE access at https://huggingface.co/settings/tokens"
                }
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"⚠️  Hugging Face timeout (model loading): {e}")
                return {
                    "success": False,
                    "error": "Model loading timeout. HF free tier may have slow startup. Try again in 30 seconds."
                }
            else:
                print(f"⚠️  Hugging Face error: {e}")
                return {
                    "success": False,
                    "error": f"HF API error: {str(e)[:100]}"
                }


if __name__ == "__main__":
    # Test the LLM provider
    print("Testing Hugging Face LLM Provider...\n")
    
    # Test Hugging Face API
    result = LLMProvider.ask_about_food(
        "Ayam Pop",
        "Berikan resep lengkap untuk membuat makanan ini"
    )
    print(f"\nHugging Face Test:")
    print(f"Success: {result['success']}")
    print(f"Provider: {result.get('provider')}")
    if result['success']:
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result.get('error')}")
    
    # Test another food
    print("\n" + "="*60)
    result = LLMProvider.ask_about_food(
        "Gulai Ikan",
        "Apa saja nutrisi dalam makanan ini?"
    )
    print(f"\nSecond Test:")
    print(f"Success: {result['success']}")
    print(f"Provider: {result.get('provider')}")
    if result['success']:
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result.get('error')}")
