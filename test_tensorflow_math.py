import tensorflow as tf
import numpy as np

def test_tensorflow_math_functions():
    """Test các hàm mũ và toán học trong TensorFlow"""
    
    print("🧪 Testing TensorFlow Math Functions")
    
    # Tạo dữ liệu test
    x = tf.constant([1.0, 2.0, 3.0, 4.0])
    y = tf.constant([2.0, 3.0, 2.0, 1.0])
    
    print(f"x = {x}")
    print(f"y = {y}")
    print()
    
    # 1. tf.pow() - Lũy thừa
    print("1. tf.pow(x, y) - Lũy thừa:")
    result = tf.pow(x, y)
    print(f"   x^y = {result}")
    print()
    
    # 2. tf.exp() - Hàm mũ tự nhiên
    print("2. tf.exp(x) - Hàm mũ tự nhiên (e^x):")
    result = tf.exp(x)
    print(f"   e^x = {result}")
    print()
    
    # 3. tf.square() - Bình phương
    print("3. tf.square(x) - Bình phương:")
    result = tf.square(x)
    print(f"   x^2 = {result}")
    print()
    
    # 4. tf.sqrt() - Căn bậc hai
    print("4. tf.sqrt(x) - Căn bậc hai:")
    result = tf.sqrt(x)
    print(f"   √x = {result}")
    print()
    
    # 5. Toán tử **
    print("5. x ** y - Toán tử lũy thừa:")
    result = x ** y
    print(f"   x**y = {result}")
    print()
    
    # 6. Test với tensor 2D
    print("6. Test với tensor 2D:")
    matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    print(f"   Matrix = {matrix}")
    print(f"   Matrix^2 = {tf.pow(matrix, 2)}")
    print(f"   e^Matrix = {tf.exp(matrix)}")
    print()
    
    # 7. Test task-aligned scores
    print("7. Test Task-Aligned Scores:")
    ious = tf.constant([[0.8, 0.6, 0.4], [0.7, 0.9, 0.3]])
    scores = tf.constant([[0.9, 0.7, 0.5], [0.8, 0.6, 0.4]])
    
    print(f"   IoUs shape: {ious.shape}")
    print(f"   Scores shape: {scores.shape}")
    
    # Task-aligned scores = IoU^alpha * Score^beta
    alpha = 1.0
    beta = 6.0
    
    aligned_scores = tf.pow(ious, alpha) * tf.pow(scores, beta)
    print(f"   Aligned scores = IoU^{alpha} * Score^{beta}")
    print(f"   Result: {aligned_scores}")
    print()
    
    # 8. Test với giá trị âm và zero
    print("8. Test với giá trị đặc biệt:")
    special_values = tf.constant([-1.0, 0.0, 1.0, 2.0])
    print(f"   Values: {special_values}")
    print(f"   e^x: {tf.exp(special_values)}")
    print(f"   x^2: {tf.square(special_values)}")
    print(f"   √x: {tf.sqrt(tf.abs(special_values))}")  # sqrt của giá trị tuyệt đối

if __name__ == "__main__":
    test_tensorflow_math_functions() 