# stt-lab

Стенд для замера онлайн-транскрибации русской речи на CPU и прототип сервиса
потокового распознавания для двух дорожек (звонок один на один). Модель — [T-one](https://github.com/voicekit-team/T-one)
(Apache-2.0, потоковая, 8 кГц, чанк 300 мс); для сравнения качества —
[GigaAM-v3](https://github.com/salute-developers/GigaAM) (MIT, пакетная, `e2e` с пунктуацией),
Vosk и Deepgram.

## Зонд за минуту: сколько потянет ваш CPU

Нужен Python 3.10–3.12, root и T-one не нужны.

```bash
git clone https://github.com/denizen24/stt-lab && cd stt-lab
python3 -m venv .venv && .venv/bin/pip install onnxruntime numpy huggingface-hub soundfile
curl -LO https://raw.githubusercontent.com/voicekit-team/T-one/main/tone/demo/audio_examples/audio_short.flac
.venv/bin/python bench/acoustic.py --audio audio_short.flac --threads 1 2 4
```

Смотрите `чанк p50` при `threads=1`: чанк длится 300 мс, значит одно ядро
тянет примерно 300 / p50 дорожек (с запасом — вдвое меньше). Для сравнения:
Xeon с AVX2 — 44 мс, Xeon E5-2680 v2 — 109–117 мс.

## Что здесь есть

- `bench/acoustic.py` — зонд: сколько стоит один чанк 300 мс на вашем CPU, а значит, сколько
  дорожек он потянет. Зависит только от onnxruntime, numpy, huggingface-hub и soundfile,
  ставится за минуту без пакета T-one.
- `bench/bench.py` — нагрузка: N дорожек реального времени, p50/p95 тика, отставание,
  CPU и RSS; режимы `batch` (один вызов на все дорожки) и `threads` (дорожка на рабочий
  поток, `intra_op=1`); декодер greedy или KenLM. `bench/run_curve.sh` снимает всю кривую,
  `bench/plot.py` рисует её.
- `bench/to_fp32.py`, `bench/quantize.py`, `bench/compare_models.py`, `bench/kenlm_probe.py` —
  эксперименты с моделью: fp32, int8, цена декодера KenLM.
- `quality/` — WER/CER по единой нормализации (`score.py`) и раннеры на одних файлах:
  `run_tone.py`, `run_gigaam.py`, `run_vosk.py`, `run_deepgram.py`.
- `service/` — `stt-streaming`: один процесс, одна модель, две WebSocket-дорожки на встречу,
  ресемплинг 48 → 8 кГц, лимит дорожек, `finalize`, `/metrics`. `client_demo.py` — пробный
  клиент из двух файлов. Контракт описан в заголовке `service/app.py`.
- `deploy/` — `install-node.sh` (пакеты, venv, веса), `install-extras.sh` (GigaAM, Vosk в
  своих venv), `sync-node.sh`, systemd-юнит с CPUQuota, Dockerfile.
- `results/` — таблицы прогонов (`bench-<host>.csv`), описание машины (`env-<host>.md`),
  журнал находок (`notes-<host>.md`), графики. Сырые тики и тексты записей в git не кладутся.
- `samples/` — аудио; в git никогда. `samples/ref/` — эталоны с пометкой, вычитка это или нет.

## Полный стенд

Нужны Ubuntu, root и Python 3.10–3.12; ставится в `/srv/stt-lab`.

```bash
sudo bash deploy/install-node.sh                     # пакеты, venv в /srv/stt-lab, веса T-one
PY=/srv/stt-lab/.venv/bin/python
$PY bench/env.py                                     # машина → results/env-<host>.md
bash bench/prepare_samples.sh                        # образцы T-one → samples/*.8k.wav
$PY bench/acoustic.py --audio samples/tone-short.8k.wav --threads 1 3   # зонд
$PY bench/bench.py --audio samples/tone-long.8k.wav --streams 2 4 6 --mode threads
```

Сервис:

```bash
$PY bench/to_fp32.py                                 # та же модель в float32: грузится втрое быстрее, вдвое меньше RSS
STT_SHARED_SECRET=… STT_MAX_STREAMS=4 STT_MODEL_PATH=$PWD/models/T-one-fp32/model.onnx \
  /srv/stt-lab/.venv/bin/uvicorn service.app:main --factory --host 127.0.0.1 --port 8090
$PY service/client_demo.py --host samples/tone-long.8k.wav --client samples/tone-short.8k.wav
$PY -m pytest -q tests/                              # контракт на подменном движке, модель не нужна
```

Ключи облачных сервисов для контрольных прогонов — в `.env` по образцу `.env.example`.

## Что показал стенд

Два CPU, один код. **A**: Xeon с AVX2, 8 vCPU — чанк 300 мс стоит 44 мс в один поток,
в 2,5 раза быстрее на ядро, чем B. **B**: Xeon E5-2680 v2 (2013, AVX без AVX2), 6 vCPU —
109–117 мс; B держит 4–5 дорожек (p95 тика < 300 мс), то есть две встречи. Режим `threads`
(дорожка на поток, `intra_op=1`) лучше `batch`: 5 дорожек против 2. KenLM почти бесплатен
по CPU, но процесс занимает 5,5 ГБ RAM против 0,6 ГБ у greedy. Подробности —
`results/notes-node-b.md`.

## Чего здесь нет

Аудио и расшифровки реальных разговоров в репозиторий не входят; сервис не пишет в логи
ни звук, ни текст.

## Лицензия

MIT, см. `LICENSE`. T-one — Apache-2.0, GigaAM — MIT; тексты в `results/quality/tone-*`
получены из демо-аудио T-one (см. `NOTICE`).
