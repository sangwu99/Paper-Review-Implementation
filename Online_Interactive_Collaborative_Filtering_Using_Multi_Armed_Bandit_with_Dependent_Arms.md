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
즉 arm의 dependency도 알아낼 수 있다는 것. 또한 잠재변수는 UCB, Thompson sampling 과 같은 multi arm selection 전략과 융합될 수 있음. 
