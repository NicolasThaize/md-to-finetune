from core.parsing import HierarchicalMarkdownParser
from core.chunking import MarkdownChunkProcessor
from exporters.messages import MessagesFormatExporter
from exporters.question_answer import QuestionAnswerFormatExporter
from exporters.user_assistant import UserAssistantFormatExporter
from exporters.mistral_template import MistralChatTemplateExporter
from llm.ollama_generators import LLMQuestionGenerator, LLMAnswerGenerator
from pipeline.generator import DatasetGenerator, QAPairGenerator


class DatasetGeneratorFactory:
    """Factory pour créer des générateurs de dataset."""
    
    @staticmethod
    def create(
        llm_model: str = "mistral:7b-instruct",
        output_format: str = "messages",
        mistral_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    ) -> DatasetGenerator:
        parser = HierarchicalMarkdownParser()
        chunker = MarkdownChunkProcessor()
        q_gen = LLMQuestionGenerator(llm_model)
        a_gen = LLMAnswerGenerator(llm_model)
        qa_generator = QAPairGenerator(q_gen, a_gen)
        
        if output_format == "messages":
            exporter = MessagesFormatExporter()
        elif output_format == "question_answer":
            exporter = QuestionAnswerFormatExporter()
        elif output_format == "user_assistant":
            exporter = UserAssistantFormatExporter()
        elif output_format == "mistral_template":
            exporter = MistralChatTemplateExporter(model_name=mistral_model_name)
        else:
            raise ValueError(
                "Format de sortie inconnu: {output_format}. "
                "Formats disponibles: messages, question_answer, user_assistant, mistral_template"
            )
        
        return DatasetGenerator(parser, chunker, qa_generator, exporter)

from exporters.messages import MessagesFormatExporter
from exporters.question_answer import QuestionAnswerFormatExporter
from exporters.user_assistant import UserAssistantFormatExporter
from exporters.mistral_template import MistralChatTemplateExporter

from core.parsing import HierarchicalMarkdownParser
from core.chunking import MarkdownChunkProcessor
from llm.ollama_generators import LLMQuestionGenerator, LLMAnswerGenerator

from .generator import DatasetGenerator, QAPairGenerator


class DatasetGeneratorFactory:
    """Factory pour créer des générateurs de dataset."""
    
    @staticmethod
    def create(
        llm_model: str = "mistral:7b-instruct",
        output_format: str = "messages",
        mistral_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ) -> DatasetGenerator:
        parser = HierarchicalMarkdownParser()
        processor = MarkdownChunkProcessor()
        qgen = QAPairGenerator(
            LLMQuestionGenerator(llm_model),
            LLMAnswerGenerator(llm_model)
        )
        if output_format == "messages":
            exporter = MessagesFormatExporter()
        elif output_format == "question_answer":
            exporter = QuestionAnswerFormatExporter()
        elif output_format == "user_assistant":
            exporter = UserAssistantFormatExporter()
        elif output_format == "mistral_template":
            exporter = MistralChatTemplateExporter(model_name=mistral_model_name)
        else:
            raise ValueError(
                "Format de sortie inconnu: {}. Formats disponibles: messages, question_answer, user_assistant, mistral_template".format(output_format)
            )
        return DatasetGenerator(parser, processor, qgen, exporter)


