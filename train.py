from src.modeler import modeler

model = modeler().create_model(summary=True).load_weight().train()
model.plot_train_history()
