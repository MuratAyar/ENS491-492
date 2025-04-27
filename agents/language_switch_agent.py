from typing import Dict, Any

class LanguageSwitchAgent:
    def __init__(self, target_language: str = "en"):
        self.target_language = target_language

    async def run(self, transcript: str) -> Dict[str, Any]:
        """Detect language of the transcript and translate if not in target language."""
        try:
            # Import here to avoid dependency if not needed
            from langdetect import detect
        except ImportError:
            return {"error": "langdetect module is not installed."}
        try:
            language = detect(transcript)
        except Exception as e:
            return {"error": f"Language detection failed: {e}"}
        result: Dict[str, Any] = {"original_language": language}
        if language and language.lower() != self.target_language.lower():
            translated_text = None
            try:
                import argostranslate.package, argostranslate.translate
                # Ensure the language codes match any installed Argos packages
                installed_langs = argostranslate.translate.get_installed_languages()
                from_lang_obj = next((lang for lang in installed_langs if lang.code.startswith(language)), None)
                to_lang_obj = next((lang for lang in installed_langs if lang.code.startswith(self.target_language)), None)
                if from_lang_obj and to_lang_obj:
                    translated_text = from_lang_obj.get_translation(to_lang_obj).translate(transcript)
            except Exception as e:
                print(f"[LanguageSwitch] Translation error or model not found: {e}")
            if translated_text:
                result["transcript"] = translated_text
                result["translation_used"] = True
            else:
                # Could not translate (no model or error), just note that language differs
                result["translation_used"] = False
        return result
