#!/usr/bin/env python
# coding: utf-8

# In[84]:


# 데이터 불러오기
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

# 학습 및 결과에 필요한 패키지들
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np


# ## 1. 데이터 준비
# - 손글씨 분류 (load_digits)
# - 와인 분류 (load_wine)
# - 유방암 분류 (load_breast_cancer)

# In[85]:


#객체 생성
digits = load_digits()
wine = load_wine()
breast_cancer = load_breast_cancer()

# 각 데이터 컬럼 확인 확인
print(digits.keys())
print(wine.keys())
print(breast_cancer.keys())


# # 손글씨 분류하기
# 
# ## 데이터 이해하기

# In[86]:


# digits의 featuredata
print("digit feature")
print(digits.data)
print(digits.data.shape)
# digits의 labeldata
print("digit label")
print(digits.target)
print(digits.target.shape)
print("digit target_names")
print(digits.target_names)

# 데이터 Describe
print("digit describe")
print(digits.DESCR)


# feature들의 정보를 이용해서 0~9까지의 숫자를 판별하는 데이터 세트  
# 1797개의 데이터수, 64 feature

# ## train, test 데이터 분리

# In[87]:


# 데이터셋 분리 훈련 데이터 0.8, 테스트 데이터 0.2의 비율
X_train, X_test, y_train, y_test = train_test_split(digits.data, 
                                                    digits.target,
                                                    test_size = 0.2,
                                                    random_state=7)

print('X_train 개수: ', X_train.shape, ", X_test 개수", X_test.shape)
print('y_train 개수: ', y_train.shape, ", y_test 개수", y_test.shape)


# ## 다양한 모델로 학습시켜보기
# 
# ### Decision Tree

# In[88]:


from sklearn.tree import DecisionTreeClassifier

# 객체 생성
decision_tree = DecisionTreeClassifier(random_state=32)
# 모델 학습
decision_tree.fit(X_train, y_train)
# 모델 예측
y_pred = decision_tree.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### RandomForest

# In[89]:


from sklearn.ensemble import RandomForestClassifier

# 객체 생성
random_forest = RandomForestClassifier(random_state=32)
# 모델 학습
random_forest.fit(X_train, y_train)
# 모델 예측
y_pred = random_forest.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### SVM

# In[90]:


from sklearn import svm

# 객체 생성
svm_model = svm.SVC(C=1, kernel='rbf')
# 모델 학습
svm_model.fit(X_train, y_train)
# 모델 예측
y_pred = svm_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ## SVC 모델 파라미터
# SVC모델을 사용할땐 오류율을 조절해줘야한다. default는 1이다.  
# 오류율 파라미터는 C 이고, C = 1 은 아웃 라이어를 하용하지 않는 하드 마진이라고 한다. 하드 마진은 마진이 적은것을 의미하고, 그래서 overfitting 문제에 빠질 수가 있다.  
# 반대로 파라미터를 C = 0 에 가깝게 설정하면 소프트 마진이라고하고, 마진이 커지게된다. 그러다보니 학습이 제대로 안되는 underfitting 문제가 발생할 수 있다.  
# 
# 
# 추가로 kernel옵션도 있다. default는 rbf이다.   
# 경계선의 모양을 여러가지로 결정할 수 있는 옵션이다. 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# ### SGD

# In[91]:


from sklearn.linear_model import SGDClassifier

# 객체 생성
sgd_model = SGDClassifier(random_state=32)
# 모델 학습
sgd_model.fit(X_train, y_train)
# 모델 예측
y_pred = sgd_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### Logistic Regression

# In[92]:


from sklearn.linear_model import LogisticRegression

# 객체 생성
logistic_model = LogisticRegression(max_iter=2100)
# 모델 학습
logistic_model.fit(X_train, y_train)
# 모델 예측
y_pred = logistic_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ## ConvergenceWarning: lbfgs failed to converge (status=1)
# 기존 예제대로 진행하다보니 위와같은 에러가 발생하였다.
# 이 에러를 찾아보니 해결방법은
# 1) iteration의 max 크기를 늘려주거나
# 2) https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression 이 링크를 따라 데이터를 조정해줘야한다고 한다.
# 
# 첫번째 방법 시도해봤다.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression
# 이 문서를 찾아보니 max_iter의 defaul 값은 100이라고 나와있다.
# 이 값을 2000까지 올렸을때는 warning이 떴으나 2100부터는 나오지 않았다.

# ## 성능평가
# 첫번째로 성능 평가의 중점을 두었던건 accuracy였다. 하지만 accuracy는 데이터의 불균형이 있을때 값을 신뢰할 수 없을 경우가 있다. 예를들어 데이터가 0이 80% 3이 20% 있을때 0으로만 판단해도 정확도는 80%가 나오기 때문이다.
# 두번째로 성능 평가를 따진것은 오차 행렬을 이용한 성능 지표들이였다.  
# precicion과 recall의 조화평균인 f1-score로 모델의 성능을 평가 하였다. 그렇게해서 가장 좋은 성능을 보인 모델은 svm 모델이였다.  
# 옳게 평가가 된것인지 궁금해서 오차행렬을 한번 프린트 해보고, 모델이 잘못 추론한 값을 살펴보았다.  
# 오차행렬의 대각행은 positive True인 행으로 옳은값을 추론한것이고, 그 외에 행은 값을 잘못 추론한 행이다.  
# 따라서 옳은값을 추론한 행을 뺀 나머지의 합을 구하면 잘못 추론한 행을 구할수 있어서 시도해 보았고, 역시나 svm모델이 제일 적게 틀린 모습을 볼수 있었다.  

# ## 와인데이터 분류하기
# ## 데이터 이해하기

# In[93]:


# wine의 featuredata
print("wine feature")
print(wine.data)
print(wine.data.shape)
# wine의 labeldata
print("wine label")
print(wine.target)
print(wine.target.shape)
print("wine target_names")
print(wine.target_names)

# 데이터 Describe
print("wine describe")
print(wine.DESCR)


# 와인의 특성들을 활용해서, 클래스 0~2를 판별하는 데이터세트  
# 178개의 데이터, 13개의 feature

# ## train, test 데이터 분리

# In[94]:


#데이터셋 분리, 훈련 데이터 0.8, 테스트 데이터 0.2 비율
X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                   wine.target,
                                                   test_size=0.2,
                                                   random_state=21)
print('X_train 개수: ', X_train.shape, ", X_test 개수", X_test.shape)
print('y_train 개수: ', y_train.shape, ", y_test 개수", y_test.shape)


# ## 다양한 모델로 학습시켜보기
# 
# ### Decision Tree

# In[97]:


# 객체 생성
decision_tree = DecisionTreeClassifier(random_state=32)
# 모델 학습
decision_tree.fit(X_train, y_train)
# 모델 예측
y_pred = decision_tree.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### RandomForest

# In[98]:


# 객체 생성
random_forest = RandomForestClassifier(random_state=32)
# 모델 학습
random_forest.fit(X_train, y_train)
# 모델 예측
y_pred = random_forest.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### SVM

# In[99]:


# 객체 생성
svm_model = svm.SVC(C=1, kernel='linear')
# 모델 학습
svm_model.fit(X_train, y_train)
# 모델 예측
y_pred = svm_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples
# 이 에러의 경우 f1-score를 구하는 과정에서 0으로 나누려 하다보니 나오는 warning 이였다.  
# 추천 옵션으로는 confusion_matrix()의 파라미터에 zero_division="warn" 인것을 0 or 1 로 바꾸라는 것이였다.  
# 하지만 위에서 알아 본것처럼 커널을 변경하면 제대로 예측을해서 이러한 문제점이 없어지진 않을까 시도해봤다.
# kernel을 linear로 바꿔보니 class 2의 예측이 0에서 11로 증가해 이 warning이 사라졌다.

# ### SGD

# In[100]:


# 객체 생성
sgd_model = SGDClassifier(random_state=32)
# 모델 학습
sgd_model.fit(X_train, y_train)
# 모델 예측
y_pred = sgd_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### SGD model의 zero_division
# 이경우도 class 2의 예측이 실패해서 나타는 경우이다.  
# sgd 모델의 파라미터값들을 살펴 보았지만. 어느것을 건들어야 성능이 좋아질지는 아직 몰라서 넘어갔다.  
# class sklearn.linear_model.SGDClassifier(loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)

# ### Logistic Regression

# In[101]:


# 객체 생성
logistic_model = LogisticRegression(max_iter=3100)
# 모델 학습
logistic_model.fit(X_train, y_train)
# 모델 예측
y_pred = logistic_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# max_iter의 값이 3100이 되서야 warning이 뜨지 않았다.

# ## 성능평가
# 수치적인 성능으로만 봤을때는 RandomForest와 logistic regression 모델이 둘다 100%의 예측을 보여주었다.  
# 그래서 학습 데이터 양과 테스트 양의 변화가 있을때 두 모델이 어떤 변화가 있는지 궁금해 아래 추가적으로 실험을 해봤다.  
# 그랬더니 randomforest의 결과 값이 우세한 모습을 보여주었다.  
# 따라서 wine data에서는 randomforest의 모델이 좋은것으로 판단된다.

# In[102]:


X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                   wine.target,
                                                   test_size=0.4,
                                                   random_state=21)
print('X_train 개수: ', X_train.shape, ", X_test 개수", X_test.shape)
print('y_train 개수: ', y_train.shape, ", y_test 개수", y_test.shape)


# In[103]:


# 객체 생성
random_forest = RandomForestClassifier(random_state=32)
# 모델 학습
random_forest.fit(X_train, y_train)
# 모델 예측
y_pred = random_forest.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# In[104]:


# 객체 생성
logistic_model = LogisticRegression(max_iter=4100)
# 모델 학습
logistic_model.fit(X_train, y_train)
# 모델 예측
y_pred = logistic_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ## 유방암 데이터 분류하기
# ## 데이터 이해하기

# In[105]:


#데이터 이해
# breast_cancer의 featuredata
print("breast_cancer feature")
print(breast_cancer.data)
print(breast_cancer.data.shape)
# breast_cancer의 labeldata
print("breast_cancer label")
print(breast_cancer.target)
print(breast_cancer.target.shape)
print("breast_cancer target_names")
print(breast_cancer.target_names)

# 데이터 Describe
print("breast_cancer describe")
print(breast_cancer.DESCR)


# 다양한 유방암 특성들을 활용해 양성인지 음성인지 판단하는 세트  
# 569개의 데이터, 30개의 feature

# In[106]:


# train test 데이터 분리
# 데이터셋 분리 훈련 데이터 0.8, 테스트 데이터 0.2의 비율
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data,
                                                    breast_cancer.target,
                                                    test_size = 0.2,
                                                    random_state=7)

print('X_train 개수: ', X_train.shape, ", X_test 개수", X_test.shape)
print('y_train 개수: ', y_train.shape, ", y_test 개수", y_test.shape)


# ### DecisionTree

# In[107]:


# 객체 생성
decision_tree = DecisionTreeClassifier(random_state=32)
# 모델 학습
decision_tree.fit(X_train, y_train)
# 모델 예측
y_pred = decision_tree.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### RandomForest

# In[108]:


# 객체 생성
random_forest = RandomForestClassifier(random_state=32)
# 모델 학습
random_forest.fit(X_train, y_train)
# 모델 예측
y_pred = random_forest.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### SVM

# In[109]:


# 객체 생성
svm_model = svm.SVC(C=0.5, kernel='linear')
# 모델 학습
svm_model.fit(X_train, y_train)
# 모델 예측
y_pred = svm_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### SVM 파라미터 바꿔보기
# 1. rbf  
# - C = 1 wrong predict : 11  
# - C = 0.5 wrong predict : 11  
# 
# 
# 2. linear
# - C = 1 wrong predict : 6  
# - C = 0.5 wrong predict : 6  
# 
# 
# default 파라미터인 rbf보다, linear의 성능이 살짝 향상 되었다.

# ### SGD

# In[110]:


# 객체 생성
sgd_model = SGDClassifier(random_state=32)
# 모델 학습
sgd_model.fit(X_train, y_train)
# 모델 예측
y_pred = sgd_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_pred, y_test))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ### LogisticRegression

# In[111]:


# 객체 생성
logistic_model = LogisticRegression(max_iter=2100)
# 모델 학습
logistic_model.fit(X_train, y_train)
# 모델 예측
y_pred = logistic_model.predict(X_test)
# 모델 성능 평가
print(classification_report(y_pred, y_test))
# 오차행렬 표시
print(confusion_matrix(y_test, y_pred))
# 대각 행렬은 정확하게 예측한곳, 그외는 잘못 예측한 계수
cm = confusion_matrix(y_pred, y_test)
print("wrong predict : ", cm.sum() - np.diag(cm).sum())


# ## 성능평가
# 유방암 데이터 셋에서 중요한건 암 환자를 정상인으로 판단하는 오류이다.   
# 유방암 환자는(malignant) 데이터에서 0을 나타내고 있고, confusion matrix에서 0번째 행을 나타낸다.  
# 따라서 유방암 환자를 정상인으로 잘못 예측한 (0,1) 요소를 잘 봐야했고. 여기서 logisticregression과 decisontree는 탈락하게 되었다.  
# 이제 남은 모델은 치명적인 예측의 실수는 없었지만. 제일 정확하게 판단한 randomforest 모델이 가장 성능이 좋았다.  
# 

# 이번 Exploration을 통해서 sklearn에 대해 좀더 알수 있었고, 같은 데이터라도 모델에 따라서 결과값이 다르고, 각 모델도 파라미터에 따라서 결과가 달라진다는 것을 알게 되어서 좋았다.
