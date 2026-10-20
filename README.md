# stt-lab

Стенд для замеров распознавания русской речи на CPU и прототип сервиса живых субтитров
для звонка один на один (две дорожки, по одной на участника). Стенд отвечает на два вопроса:

1. **Черновик в реальном времени.** Сколько дорожек потянет конкретный CPU с потоковой моделью
   [T-one](https://github.com/voicekit-team/T-one) (Apache-2.0, 8 кГц, чанк 300 мс) и насколько
   она уступает [GigaAM-v3](https://github.com/salute-developers/GigaAM) (MIT), Vosk и облачному Deepgram.
2. **Чистовик после звонка.** Что даёт второй проход пакетной моделью (GigaAM-v3 и Parakeet TDT 0.6B v3
   через [onnx-asr](https://github.com/istupakov/onnx-asr), fp32/int8), каскад двух моделей и личный
   словарь, и можно ли гонять чистовик на той же машине, где идут субтитры.

Код один для обеих статей; README описывает текущее состояние, команды сверены с ним.
Репозиторий: [GitHub](https://github.com/denizen24/stt-lab) · [GitVerse](https://gitverse.ru/denizen24/stt-lab)
(зеркало, содержимое одинаковое).

## Статьи

- **Часть 1.** [Открытые модели против облака: T-one, GigaAM или Deepgram: что реально тянет обычный CPU
  для живых субтитров в звонке](https://habr.com/ru/articles/1095420/) — сравнение T-one, GigaAM, Vosk и
  Deepgram на CPU против облака, сервис живых субтитров. Повторить — [ниже](#повторить-первую-часть).
- **Часть 2** *(готовится; ссылка появится после публикации)*. Второй проход «черновик → чистовик» и
  каскад моделей (задача 121-474). Повторить — [ниже](#повторить-вторую-часть).

## Что здесь есть

- `bench/` — скорость: `acoustic.py` (зонд: цена одного чанка 300 мс), `bench.py` (N дорожек реального
  времени, p50/p95 тика, отставание, CPU, RSS; режимы `batch` и `threads`; декодер greedy или KenLM),
  `run_curve.sh` + `plot.py` (кривая «дорожки → тик»), `to_fp32.py`, `quantize.py`, `compare_models.py`,
  `kenlm_probe.py` (эксперименты с моделью), `env.py` (описание машины), `prepare_samples.sh`
  (аудио → моно 8 или 16 кГц). Серии прогонов: `run_quality_set.sh` (часть 1), `run_batch_set.sh`,
  `run_etalon_set.sh`, `run_corun.sh` (часть 2).
- `quality/` — качество: раннеры движков, WER/CER, метрики сверх WER, каскад, личный словарь, списки
  спорных мест для вычитки. Все скрипты — в разделе [«Папка quality/»](#папка-quality).
- `service/` — `stt-streaming`: один процесс, одна модель, две WebSocket-дорожки на встречу, ресемплинг
  48 → 8 кГц, лимит дорожек, `finalize`, `/metrics`. `client_demo.py` — пробный клиент из двух файлов.
  Контракт описан в заголовке `service/app.py`.
- `deploy/` — `install-node.sh` (пакеты, venv, веса T-one), `install-extras.sh` (GigaAM и Vosk в своих
  venv), `install-onnxasr.sh` (onnx-asr и Parakeet для второй части), `sync-node.sh`, systemd-юнит
  с CPUQuota, Dockerfile.
- `results/` — таблицы прогонов: `bench-<хост>.csv` (скорость T-one), `batch-<хост>.csv` (пакетные
  прогоны onnx-asr), `env-<хост>.md` (машина), `notes-<хост>.md` (журнал находок), графики.
- `samples/` — аудио и эталоны (`samples/ref/`); в git не попадают вовсе.

## Быстрый старт

### Зонд за минуту: сколько потянет ваш CPU

Нужен Python 3.10–3.12; root и пакет T-one не нужны.

```bash
git clone https://github.com/denizen24/stt-lab && cd stt-lab     # или https://gitverse.ru/denizen24/stt-lab
python3 -m venv .venv && .venv/bin/pip install onnxruntime numpy huggingface-hub soundfile
curl -LO https://raw.githubusercontent.com/voicekit-team/T-one/main/tone/demo/audio_examples/audio_short.flac
.venv/bin/python bench/acoustic.py --audio audio_short.flac --threads 1 2 4
```

Смотрите `чанк p50` при `threads=1`: чанк длится 300 мс, значит одно ядро тянет примерно
300 / p50 дорожек (с запасом — вдвое меньше). Для сравнения: Xeon с AVX2 — 44 мс, Xeon E5-2680 v2 — 109–117 мс.

### Установка стенда

Нужны Ubuntu, root, Python 3.10–3.12 и ffmpeg (ставит `install-node.sh`).
Окружения ставятся в `/srv/stt-lab` (другой каталог — переменная `STT_LAB_ROOT`; `run_batch_set.sh` и
`run_curve.sh` вместо неё берут путь к Python из `PY`), команды ниже выполняются из корня клона.

```bash
sudo bash deploy/install-node.sh                     # пакеты, venv /srv/stt-lab/.venv, T-one, веса
# sudo STT_LAB_KENLM=1 bash deploy/install-node.sh   # плюс kenlm.bin (5,4 ГБ) в фоне — для декодера KenLM
PY=/srv/stt-lab/.venv/bin/python
/srv/stt-lab/.venv/bin/pip install -r quality/requirements.txt -r service/requirements.txt onnx matplotlib
mkdir -p results/raw samples/ref                     # журналы серий и эталоны; в git не входят
```

Строка с pip дополняет `install-node.sh`: `num2words` нужен нормализации чисел в `score.py`
(без него цифры не переводятся в слова, и WER выходит другим), `service/requirements.txt` — сервису,
клиенту и тестам, `onnx` — `bench/to_fp32.py`, `matplotlib` — `bench/plot.py`.

Движки для сравнения качества, каждый в своём venv:

```bash
bash deploy/install-extras.sh                        # GigaAM-v3 (torch, .venv-gigaam) и Vosk (.venv-vosk + модели)
bash deploy/install-onnxasr.sh                       # часть 2: onnx-asr (.venv-onnxasr) и Parakeet, около 3 ГБ
```

[`deploy/install-onnxasr.sh`](deploy/install-onnxasr.sh) создаёт `/srv/stt-lab/.venv-onnxasr` и скачивает
Parakeet TDT 0.6B v3 в `models/onnx/parakeet-v3` — туда, где его ждут серийные скрипты (другой каталог —
переменная `PARAKEET`, `venv` первым аргументом — без модели). Повторный запуск безопасен. GigaAM-v3 для
onnx-asr скачивается в кэш Hugging Face сам при первом запуске; Parakeet передаётся через `--path`.

### Подготовка сэмплов

```bash
bash bench/prepare_samples.sh                                  # образцы T-one → samples/tone-{short,long}.8k.wav
bash bench/prepare_samples.sh samples/raw/запись.webm          # стерео → samples/запись.{host,client}.8k.wav
RATE=16000 bash bench/prepare_samples.sh samples/raw/запись.webm   # то же в 16 кГц → *.16k.wav (часть 2)
```

- Без аргументов скрипт берёт образцы из установленного пакета T-one (Python ищет в `.venv`, затем в
  `/srv/stt-lab/.venv`, или берёт `PY` из окружения).
- Стерео раскладывается на дорожки: левый канал — ведущий (`host`), правый — клиент (`client`). Моно
  остаётся одним файлом с тем же именем.
- `RATE=16000` — полная полоса для пакетных моделей второй части; без `RATE` — 8 кГц, как видит T-one.
- Общая дорожка `mix` (обе стороны вместе) для `bench/run_etalon_set.sh` — это моно-файл с `.mix` в имени:
  `ffmpeg -i запись.webm -ac 1 samples/raw/запись.mix.wav`, затем `prepare_samples.sh` с ним (8 и 16 кГц).

## Повторить первую часть

Все команды — из корня клона, `PY=/srv/stt-lab/.venv/bin/python`.

### Скорость

```bash
$PY bench/env.py                                     # машина → results/env-<хост>.md
bash bench/prepare_samples.sh                        # образцы T-one → samples/*.8k.wav
$PY bench/acoustic.py --audio samples/tone-short.8k.wav --threads 1 3          # зонд
$PY bench/bench.py --audio samples/tone-long.8k.wav --streams 2 4 6 --mode threads
nohup nice -n 10 bash bench/run_curve.sh > results/raw/curve.log 2>&1 &        # вся кривая, около получаса
$PY bench/plot.py --host "$(hostname)"               # results/bench-<хост>.csv → график
```

### Сервис

```bash
$PY bench/to_fp32.py                                 # модель в float32 → models/T-one-fp32/model.onnx
export STT_SHARED_SECRET=…                           # один и тот же секрет у сервиса и клиента
STT_MAX_STREAMS=4 STT_MODEL_PATH=$PWD/models/T-one-fp32/model.onnx \
  /srv/stt-lab/.venv/bin/uvicorn service.app:main --factory --host 127.0.0.1 --port 8090 &
$PY service/client_demo.py --host samples/tone-long.8k.wav --client samples/tone-short.8k.wav
$PY -m pytest -q tests/                              # контракт на подменном движке, модель не нужна
```

Ключи облачных сервисов для контрольных прогонов — в `.env` по образцу `.env.example`.

### Качество на образце T-one (51 с колл-центра)

Эталон `samples/ref/tone-long.txt` в git не входит (это согласие движков, а не вычитка) — положите свой.
Тексты движков из статьи лежат в `results/quality/tone-*`.

```bash
$PY quality/run_tone.py --decoder greedy samples/tone-short.8k.wav samples/tone-long.8k.wav
$PY quality/run_tone.py --decoder kenlm  samples/tone-short.8k.wav samples/tone-long.8k.wav   # нужен kenlm.bin
/srv/stt-lab/.venv-gigaam/bin/python quality/run_gigaam.py --model v3_e2e_rnnt samples/tone-*.8k.wav
/srv/stt-lab/.venv-gigaam/bin/python quality/run_gigaam.py --model v3_rnnt     samples/tone-*.8k.wav
/srv/stt-lab/.venv-vosk/bin/python quality/run_vosk.py --model vosk-model-ru-0.42       samples/tone-*.8k.wav
/srv/stt-lab/.venv-vosk/bin/python quality/run_vosk.py --model vosk-model-small-ru-0.22 samples/tone-*.8k.wav
$PY quality/run_deepgram.py --env .env --model nova-3 samples/tone-long.8k.wav
$PY quality/run_deepgram.py --env .env --model nova-2 samples/tone-long.8k.wav
$PY quality/score.py --table results/quality
```

### Живой диалог: две дорожки

```bash
bash bench/prepare_samples.sh samples/raw/запись.webm          # → samples/запись.{host,client}.8k.wav
nohup bash bench/run_quality_set.sh запись > results/raw/quality-запись.log 2>&1 &
#   A. сервис в реальном времени (задержки фраз)  B. T-one greedy и KenLM офлайн
#   C. GigaAM-v3 e2e  D. Vosk ru-0.42 — часа три на 30 минут аудио на слабом CPU
$PY quality/score.py --table results/quality                   # против samples/ref/запись.{host,client}.txt
$PY quality/diff_review.py --name запись.client --engines tone-kenlm gigaam-v3_e2e_rnnt vosk-ru-0.42
$PY quality/sample_review.py --record запись --roles host client --engines tone-kenlm gigaam-v3_e2e_rnnt --per-role 20 --seed 1
```

Как вычитывать по спискам спорных мест — в [`quality/README.md`](quality/README.md).

### Что изменилось с публикации первой части

Сами скрипты вызываются с теми же флагами. Изменились подготовка окружения и одна деталь подсчёта:

- **Установка:** добавлена строка `pip install -r quality/requirements.txt -r service/requirements.txt onnx
  matplotlib`. `install-node.sh` не ставит `num2words`, зависимости сервиса и клиента (`fastapi`, `uvicorn`,
  `httpx`, `websockets`, `pytest`), `onnx` для `to_fp32.py` и `matplotlib` для `plot.py`.
- **Сервис:** `STT_SHARED_SECRET` теперь экспортируется, чтобы его видел и `client_demo.py` (он шлёт
  заголовок `X-STT-Secret` из той же переменной); uvicorn запускается в фоне (`&`), чтобы клиент шёл
  из того же терминала.
- **Серии в фоне:** перед `nohup … > results/raw/…` нужен `mkdir -p results/raw` — каталог не в git.
- **`score.py`:** нормализация понимает порядковые Deepgram («1-ый» → «первый»). На живом диалоге WER всех
  движков ниже на 0,3–0,7 п.п., чем в статье; порядок движков тот же (`results/notes-node-b.md`, §6).
- **`prepare_samples.sh`:** без `RATE` работает как раньше (8 кГц); `RATE=16000` — новый режим для части 2.
- **`run_deepgram.py`:** из имени результата отбрасывается и `.16k`; вызов прежний.

## Повторить вторую часть

Окружения — из [установки](#установка-стенда), включая `deploy/install-onnxasr.sh` (onnx-asr и Parakeet).
Дорожки нужны в двух частотах: `samples/<имя>.<дорожка>.8k.wav` (черновик T-one) и `.16k.wav` (чистовик).
Серийные скрипты ждут, пока на машине нет живого звонка: если у coturn есть релейные сокеты и трафик
выше `BUSY_KBPS` (по умолчанию 50 КБ/с), прогон откладывается на 5 минут. На машине без coturn они
идут сразу.

### Чистовик через onnx-asr: GigaAM-v3 и Parakeet, fp32/int8

```bash
OX=/srv/stt-lab/.venv-onnxasr/bin/python
$OX quality/run_onnxasr.py --model gigaam-v3-e2e-rnnt --quant fp32 samples/запись.host.16k.wav samples/запись.client.16k.wav
$OX quality/run_onnxasr.py --model gigaam-v3-e2e-rnnt --quant int8 samples/запись.host.16k.wav samples/запись.client.16k.wav
$OX quality/run_onnxasr.py --model nemo-parakeet-tdt-0.6b-v3 --quant int8 --path models/onnx/parakeet-v3 \
    samples/запись.host.16k.wav samples/запись.client.16k.wav
# вся серия (GigaAM fp32/int8 на 16 и 8 кГц, Parakeet int8/fp32 на 16 кГц):
nohup bash bench/run_batch_set.sh запись > results/raw/batch-запись.log 2>&1 &
```

Флаги `run_onnxasr.py`: `--model` (`gigaam-v3-e2e-rnnt`, `gigaam-v3-rnnt`, `gigaam-v3-e2e-ctc`,
`nemo-parakeet-tdt-0.6b-v3`), `--quant` (`fp32` или `int8`), `--threads` (потоки ORT, 0 — все ядра),
`--path` (каталог модели), `--tag` (пометка строки в CSV). Нарезка по тишине та же, что в `run_gigaam.py`.
Пишет `results/quality/<имя>.<метка>.txt`, рядом `.segments.jsonl` (куски с границами — для каскада) и
строку замера в `results/batch-<хост>.csv`. Метка: `onnx-gigaam-e2e-rnnt`, `parakeet-v3` и т. п., плюс
`-int8` и `-8k`, если вход был 8 кГц.

### Эталон по сегментам

Общий эталон записи — JSON `{"segments": [{"start", "end", "speaker", "text"}, …]}` (секунды от начала).
`refs_from_segments.py` раскладывает его по дорожкам по громкости 16 кГц-файлов: сегмент уходит туда,
где он громче; если разница меньше `--margin` дБ (по умолчанию 6) — в обе. Спорные `[?вар1/вар2]`
заменяются первым вариантом, `[неразборчиво]` выбрасывается. Нужен Python не новее 3.12 (модуль `audioop`).

```bash
$PY quality/refs_from_segments.py эталон.json call-a --host-speaker "Имя ведущего"
#   → samples/ref/call-a.host.txt и samples/ref/call-a.client.txt (каталог samples/ref должен существовать)
nohup bash bench/run_etalon_set.sh call-a call-b > results/raw/etalon-set.log 2>&1 &   # дорожка mix
TRACKS="host client" bash bench/run_etalon_set.sh call-a                               # по дорожкам
```

`run_etalon_set.sh` гонит по каждой записи T-one greedy и KenLM (8 кГц), GigaAM-v3 e2e fp32 и Parakeet v3
int8 (16 кГц) с пометкой `etalon`. `refs_from_segments.py` пишет эталоны только для `host` и `client`;
для дорожки `mix` эталон `samples/ref/<имя>.mix.txt` (весь текст записи) готовится отдельно.

### Метрики сверх WER

```bash
$PY quality/metrics.py --table results/quality                                       # WER, WERx, числа, термины, пунктуация
$PY quality/metrics.py --draft-final запись.client tone-kenlm onnx-gigaam-e2e-rnnt   # черновик → чистовик
```

- **WERx** — WER «по звучанию»: латиница сводится к кириллическому ключу (Cloud-MD → «клауд эм ди»).
- **числа** — сколько числительных эталона записано не так; **термины** — то же по списку
  `samples/ref/<запись>.<роль>.terms.txt` (слово или фраза в строке).
- **пунктуация F1** — конец фразы (`. ! ?`) и запятая против эталона, только на верно распознанных словах.
- `--draft-final` — по каждому слову эталона: исправлено, испорчено, верно или неверно в обоих, и сколько
  текста поменяется у пользователя, когда чистовик заменит черновик. Тексты берутся из `--dir`
  (по умолчанию `results/quality`).

### Каскад и личный словарь

```bash
$PY quality/cascade.py запись.client onnx-gigaam-e2e-rnnt parakeet-v3-int8                  # латиница из Parakeet
$PY quality/cascade.py запись.client onnx-gigaam-e2e-rnnt parakeet-v3-int8 --dict словарь.txt
$PY quality/cascade.py запись.client onnx-gigaam-e2e-rnnt parakeet-v3-int8 --any            # контроль: любые слова
$PY quality/dict_fix.py запись.client onnx-gigaam-e2e-rnnt OpenSpec                         # без второй модели
$PY quality/metrics.py --table results/quality                                            # сравнить варианты
```

- `cascade.py` — схема pahaz (habr 1095512). Обе модели должны пройти `run_onnxasr.py` по одному файлу
  (одинаковая нарезка, иначе скрипт остановится). Слово GigaAM меняется на латинское слово Parakeet, если
  похожесть по звучанию не ниже `--threshold` (0,4). `--dict` — брать латиницу только из словаря (слова
  через пробел или по строкам, слова с `#` пропускаются), `--any` — снять требование латиницы. Пишет
  `results/quality/<запись>.cascade[-any][-dict].txt` и список замен в `.log`.
- `dict_fix.py` — термины перечисляются после меток (`OpenSpec [термин …]`). Термин заменяет слово
  чистовика, если ключ слова по звучанию начинается с ключа термина, а хвост не длиннее `--tail` букв (3).
  Пишет `<запись>.dictfix.txt` и `.dictfix.log`.

### Два прохода на одной машине

```bash
LEN=600 nohup bash bench/run_corun.sh запись > results/raw/corun-запись.log 2>&1 &
```

Берутся первые `LEN` секунд (по умолчанию 600) дорожек `запись.{host,client}.8k.wav` и `.16k.wav`.
Две встречи (четыре живых потока) через сервис в трёх сценариях: `solo`, `corun-all` (плюс GigaAM int8
на nice 19, ORT на все ядра) и `corun-t2` (ORT на 2 потока). Отставание фраз печатает `client_demo.py`,
RTF чистовика под нагрузкой — строки `results/batch-<хост>.csv` с тегами `corun-all` и `corun-t2`.
Журналы сервиса — `results/raw/corun-service-<сценарий>.log`. Нужна `models/T-one-fp32/model.onnx`
(`bench/to_fp32.py`).

### Где результаты

- `results/notes-node-b.md`, раздел 6 «Второй проход и каскад (121-474)»: качество и цена чистовика (6.1),
  черновик → чистовик (6.2), каскад (6.3), два прохода на одной машине (6.4).
- `results/batch-node-b.csv` — скорость всех пакетных прогонов (теги `set`, `corun-*`, `etalon`).
- Тексты в `results/quality/` от живых записей в git не входят (кроме образцов T-one `tone-*`).

## Папка quality/

| скрипт | что делает | venv |
|---|---|---|
| `score.py` | WER/CER по единой нормализации: пара `--ref`/`--hyp` или `--table <каталог>` | `.venv` |
| `metrics.py` | WERx, числа, термины, пунктуация F1 (`--table`), черновик → чистовик (`--draft-final`) | `.venv` |
| `run_tone.py` | T-one офлайн тем же потоковым пайплайном, `--decoder` greedy или kenlm | `.venv` |
| `run_gigaam.py` | GigaAM-v3 на torch, `--model` v3_e2e_rnnt, v3_rnnt, v3_ctc, v3_e2e_ctc; нарезка по тишине до 25 с | `.venv-gigaam` |
| `run_onnxasr.py` | GigaAM-v3 и Parakeet v3 через onnx-asr, fp32/int8, замер в `batch-<хост>.csv` | `.venv-onnxasr` |
| `run_vosk.py` | Vosk, `--model`, `--models-dir` (по умолчанию `/srv/stt-lab/models`) | `.venv-vosk` |
| `run_deepgram.py` | Deepgram pre-recorded, `--model`, `--language`, `--env` (ключ `DEEPGRAM_API_KEY`) | `.venv` |
| `refs_from_segments.py` | эталон по дорожкам из JSON с сегментами, `--margin`, `--host-speaker` | `.venv` |
| `cascade.py` | каскад правок GigaAM + Parakeet, `--threshold`, `--dict`, `--any` | `.venv` |
| `dict_fix.py` | личный словарь поверх чистовика без второй модели, `--tail` | `.venv` |
| `diff_review.py` | список спорных мест для вычитки → `<запись>.review.md` | `.venv` |
| `sample_review.py` | случайная выборка спорных мест для проверки на слух → `<запись>.sample.md` | `.venv` |

`requirements.txt` — `jiwer` и `num2words` для подсчёта. Как считаем и как вычитываем — в
[`quality/README.md`](quality/README.md).

## Что показал стенд

**Часть 1.** Два CPU, один код. **A**: Xeon с AVX2, 8 vCPU — чанк 300 мс стоит 44 мс в один поток,
в 2,5 раза быстрее на ядро, чем B. **B**: Xeon E5-2680 v2 (2013, AVX без AVX2), 6 vCPU — 109–117 мс;
B держит 4–5 дорожек (p95 тика < 300 мс), то есть две встречи. Режим `threads` (дорожка на поток,
`intra_op=1`) лучше `batch`: 5 дорожек против 2. KenLM почти бесплатен по CPU, но процесс занимает
5,5 ГБ RAM против 0,6 ГБ у greedy. Подробности — `results/notes-node-b.md`.

**Часть 2** (машина B, 29 минут живого диалога, эталон — Deepgram nova-3, не вычитка). Чистовик GigaAM-v3
e2e fp32 через onnx-asr: WER клиента 6,9% против 11,2% у T-one живьём, около 15 минут счёта на час аудио,
1,6–1,7 ГБ RAM. int8 по качеству как fp32, но на CPU без AVX2 медленнее; выигрыш только в памяти.
Замена черновика чистовиком исправляет 6,6% слов клиента и портит 1,6%. Каскад с Parakeet на этой
записи ничего не дал (латиницы почти нет), без фильтра латиницы — хуже на 1,5 п.п. Субтитры и чистовик
на одной машине уживаются: отставание фраз не изменилось, чистовик под нагрузкой идёт с RTF 0,52–0,68.
Подробности — `results/notes-node-b.md`, раздел 6.

## Чего здесь нет

Аудио и расшифровки реальных разговоров в репозиторий не входят; сервис не пишет в логи
ни звук, ни текст.

## Лицензия

MIT, см. `LICENSE`. T-one — Apache-2.0, GigaAM — MIT; тексты в `results/quality/tone-*`
получены из демо-аудио T-one (см. `NOTICE`).
