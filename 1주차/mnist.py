from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from IPython.display import SVG
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_self_attention import SeqSelfAttention

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


network = models.Sequential()#케라스의 선형모델을 생성함
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))#케라스의 레이어하나추가, inputshape = 입력형상
'''dense layer는 입력과 출력을 모두연결해준다.
첫번째 dense는 입력단
두번째는 출력단
최초 dense레이어에만 입력수 필요, 이후의 입력수는 자동으로 전 레이어의 출력수
따라서 두번쩨레이어의 10은 출력수를 의미
dense layer는 출력층으로 자주사용됨 왜냐? 입력뉴런수랑 상관없이 출력 뉴런수를 정할수있으므로?
'''
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))#활성화함수,
# 활성화 함수란 어떠한 신호를 입력받아 이를 적절한 처리를 하여 출력해주는 함수
#예제에서는 케라스에서 제공하는 relu, 소프트맥스함수를이용.
#relu함수는 선형함수,

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',#손실함수, 데이터와 추정치의 괴리도를 나타냄,값이클수록 많이틀렸다는뜻,크로스엔트로피는 대표적인 손실함수
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
#normalization 데이터셋들의 범위를 0~1로 맞춰주기 위하여 rgb값의 0~255 범위에서 가장큰 255로 나눠줌
# scale을 통일시키면 gradient를 태울때 문제가 발생하지않는다. gradient란?, feature의 범위가 너무 크면 train하기 힘들어지고 속도도 느려짐
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)#클래스 백터를 바이너리 클래스 메트릭스로변환하는함수

network.fit(train_images, train_labels, epochs=10, batch_size=128)#학숩시작 epoghs란 학습횟수 batchsize는 데이터셋 하나의 크기를 결정한다.
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

network.save('mnist1.h5')
