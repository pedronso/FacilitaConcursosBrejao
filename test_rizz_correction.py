import unittest
from pipelines.rag import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    """Testes para a função corrigir_concurso() da classe RAGPipeline."""

    def setUp(self):
        """Configura o ambiente para cada teste."""
        self.rag = RAGPipeline()

    def test_correcoes_concurso(self):
        """Testa se os nomes dos concursos são corrigidos corretamente."""
        perguntas_teste = {
            "Qual é o cronograma completo do processo seletivo da funal?": 
            "Qual é o cronograma completo do processo seletivo da funai?",

            "Qual é o cronograma completo do processo seletivo do ibga?": 
            "Qual é o cronograma completo do processo seletivo do ibge?",

            "Qual é o cronograma completo do processo seletivo do tribunal regional federau?": 
            "Qual é o cronograma completo do processo seletivo do trf?",

            "Qual é o cronograma completo do processo seletivo do ibgu?": 
            "Qual é o cronograma completo do processo seletivo do ibge?",

            "Qual é o cronograma completo do processo seletivo do concurso da energia nuclear?": 
            "Qual é o cronograma completo do processo seletivo do cnen?",

            "Qual é o cronograma completo do processo seletivo do ccev?": 
            "Qual é o cronograma completo do processo seletivo do cceb?",

            "Qual é o cronograma completo do processo seletivo do força aereaa?": 
            "Qual é o cronograma completo do processo seletivo da aeronautica?",

            "Qual é o cronograma completo do processo seletivo do agencia espacial?": 
            "Qual é o cronograma completo do processo seletivo da aeb?",
        }

        for pergunta, esperado in perguntas_teste.items():
            with self.subTest(pergunta=pergunta):
                resposta = self.rag.corrigir_concurso(pergunta)
                self.assertEqual(resposta, esperado, f"Erro na correção: {pergunta} → {resposta}")

if __name__ == "__main__":
    unittest.main()
