def etapa_4_testar_rag(local=False):
    """Executa um teste no chatbot RAG."""
    print("\n🤖 [4/6] Testando consultas ao sistema RAG...")
    
    rag = RAGPipeline()
    queries_teste = [
        "Quais são os concursos para engenheiros?",
        "Quais os prazos de inscrição?",
        "Tem algum concurso para nível médio?",
        "Quantas vagas estão abertas para o IBAMA?",
        "O MPU está com concursos abertos?"
    ]
    
    for query in queries_teste:
        try:
            resposta = rag.generate_full_answer(query)
            print(f"\n🔹 Pergunta: {query}")
            print(f"💬 Resposta: {resposta}")
        except Exception as e:
            print(f"❌ Erro ao gerar resposta para '{query}': {e}")