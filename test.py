import numpy as np
from keras.models import load_model
model = load_model("digit_model.h5")
test_img = np.random.rand(1, 28, 28, 1)  # Fake test image
print(model.predict(test_img))  # Should output 10 probabilities