def save_model(model, file_path):
    model.save_weights(file_path)

def load_model(model, file_path):
    model.load_weights(file_path)
