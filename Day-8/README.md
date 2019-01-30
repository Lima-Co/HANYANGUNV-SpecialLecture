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
