# Logistic Regression

## 참과 거짓을 나타내는 함수

![logisticGraph1](/images/logisticGraph1.png)

- y = wx 함수로 표현이 불가능 하다.

#### sigmoid 함수

![logisticGraph2](/images/logisticGraph2.png)

- y = 1/1+e^-wx 로 표현

```python
hypothesis = tensorflow.sigmoid(tensorflow.matmul(X, W) + b)
```

### sigmoid 신경 세포 기능
![sigmoid](/images/sigmoid.png)
- 입력 x에 따라 0, 혹은 1(fire)을 출력함
