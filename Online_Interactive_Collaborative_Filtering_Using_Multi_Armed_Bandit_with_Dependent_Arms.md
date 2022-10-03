# Section 1

온라인 상호작용 추천 시스템 알고리즘은 적절한 아이템 추천과 유저의 기대 만족 극대화를 위한 목표를 가짐

이 과정에서 두 가지 문제를 해결해야 함. Cold start problem, 모든 가능한 정보를 가용하지 못하는 문제

## Cold Start Problem

Exploration(새로운 정보를 얻는) Explotation (가능한 높은 효용을 취하는) 딜레마를 가짐 이는 흔히 multi armed-bandit 으로 공식화 됨

## 정보 가용 문제

CF로 해결

## Online Interactive CF

Multi Bandit과 CF를 모두 채택

잠재 변수를 만드는 행렬 분해에 multi bandit을 접목시켜 잠재변수를 update (다른 선택과 독립을 가정)

그러나 현실 세계에서 독립성 가정은 흔히 실패함
장기 만족 극대화에 독립적이지 않음이 영향을 미치는 것 

## 본 논문

본 논문에서는 잠재 변수를 학습하고 잠재 상태를 추론하는 시계열 온라인 추론법을 사용

arm의 cluster까지 arm의 선택이 영향을 미침
잠재 cluster는 동일하며 cluster의 다른 arm의 reward를 포착할 수 있음.
즉 arm의 dependency도 알아낼 수 있다는 것. 또한 잠재변수는 UCB, Thompson sampling 과 같은 multi arm selection 전략과 융합될 수 있음

# Section 2 

선행 연구 검토

## Interactive Collaborative Filtering 

다년간 유용한 방법이지만, Cold Start Problem에 취약함 기존에는 프로필을 명확화 하여 문제를 해결하려 함 
반면 ICF라는 방법론이 부상중인데, Multi-bandit 문제화 하여 Exploration, Explotation으로 우리가 주로 사용할 방법 

그러나 ICF는 밴딧 알고리즘이 Latent vector 기반으로 학습하기에 완전 Online에서는 제대로 가용하지 못함 
이에 PTS 알고리즘은 ICF Problem을 베이지안 matrix factorization 으로 해결함

그러나 이 방법은 arm의 dependency를 고려하지 않음 이를 우리가 해결할 것! 

왕 교수님은 유저간 관련성을 이용하는 방법을 채택했지만, 우리는 이와 직교적으로 item 간의 종속성을 공식화 할 것

우리는 명시된 메타데이터 (정보) 없이 Online 에서 항목 의존성을 학습할 수 있음

Multi-bandit의 Exploration Explotation Problem은 여러가지가 있으나 UCB와 Thompson sampling 으로 이를 해결할 것 

## Sequential Online Inference 

arm간의 종속성을 공식화하기 위해 우리는 Sequential Monte Carlo Sampling을 사용할 것
이는 비선형, 비 가우시안 공간 모델의 후방 분포를 추론할 수 있게 함

우리는 Particle Learning을 통해 Latent space와 parameter learning을 해결할 것 

# Section 3 

<img width="500px" alt='캡쳐1' src = "https://ieeexplore-ieee-org.proxy.cau.ac.kr/mediastore_new/IEEE/content/media/69/8755525/8440090/li.t1-2866041-small.gif">
