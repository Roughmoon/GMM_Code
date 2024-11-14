import tensorflow as tf
from tensorflow.keras import layers, models

# 加载数据
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化数据

# 定义模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 输入层，将28x28的图像转换成784维的向量
    layers.Dense(512, activation='relu'),  # 第一个隐藏层，512个节点
    layers.Dropout(0.3),  # Dropout层，防止过拟合
    layers.Dense(256, activation='relu'),  # 第二个隐藏层，256个节点
    layers.Dropout(0.3),  # Dropout层，防止过拟合
    layers.Dense(128, activation='relu'),  # 第三个隐藏层，128个节点
    layers.Dropout(0.3),  # Dropout层，防止过拟合
    layers.Dense(64, activation='relu'),  # 第四个隐藏层，64个节点
    layers.Dropout(0.3),  # Dropout层，防止过拟合
    layers.Dense(10)  # 输出层，10个节点对应10个类别
])

# 编译模型
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# 预测新数据
predictions = model.predict(x_test)