def generate_metrics(description, num_samples=3):
  """ choose metrics based on the description """
  list_metrics = ['precision', 'recall', 'toxicity', 'hallucination', 'relevancy', 'bias', 'knowledge retention']
  
  PROMPT_TEMPLATE = f"""You are auditing AI agents. You only audit the content the is generated by the agent. You must take the description of the AI agent and generate suitable evaluation metrics for its generated content.

  The agent to be audited:
  - description of the agent: {description}
  - list of metrics to choose from: {list_metrics}
  - number of metrics to generate: {num_samples}

  You must choose top{num_samples} metrics from the list of metrics provided that are suitable for evaluating the agent. No explaination needed!
  """

  metrics_chosen = mistral(PROMPT_TEMPLATE, model).split(',')
  return metrics_chosen
