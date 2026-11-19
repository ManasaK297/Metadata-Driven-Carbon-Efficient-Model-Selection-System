"""Train and save meta-model from knowledge base."""

from src.model_selector import ModelSelector

print("Training meta-model from knowledge base...")
selector = ModelSelector('knowledge_base.csv')
selector.train_meta_model()
selector.save_meta_model('meta_model.pkl')
print('Meta-model trained and saved to meta_model.pkl')
