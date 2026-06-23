from agents.health_agent import HealthAgent

agent = HealthAgent()

answer, docs = agent.generate_response(
    "Can I drink coffee?"
)

print(answer)