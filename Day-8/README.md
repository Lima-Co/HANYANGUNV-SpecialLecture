# 8일차 챗봇 실습 및 종합

## 학습 데이터 소개

### 단어집 생성

#### 질문 단어집

```bash
python generate_vocab.py < data\train.req > data\vocab.req
```

정상적으로 완료되었다면 data 디렉토리 하위에 **vocab.req**라는 파일이 생성됩니다.

#### 답변 단어집

```bash
python generate_vocab.py < data\train.rep > data\vocab.rep
```

정상적으로 완료되었다면 data 디렉토리 하위에 **vocab.rep**라는 파일이 생성됩니다.

## 데이터 학습

### 학습 명령어

```bash
python -m nmt.nmt --src=req --tgt=rep --vocab_prefix=data/vocab --train_prefix=data/train --dev_prefix=data/dev --test_prefix=data/test --out_dir=nmt_model --num_train_steps=12000 --steps_per_stats=100 --num_layers=4 --num_units=128 --dropout=0.2 --metrics=bleu --attention=scaled_luong
```

학습 시간을 줄이기 위해 __학습 횟수(num_train_steps)__를 12000에서 3000으로 줄였습니다.

```bash
python -m nmt.nmt --src=req --tgt=rep --vocab_prefix=data/vocab --train_prefix=data/train --dev_prefix=data/dev --test_prefix=data/test --out_dir=nmt_model --num_train_steps=3000 --steps_per_stats=100 --num_layers=4 --num_units=128 --dropout=0.2 --metrics=bleu --attention=scaled_luong
```
