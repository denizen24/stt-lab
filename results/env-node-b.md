# Машина стенда: node-b (машина B)

Снято: 2026-10-16T14:09+03:00

| Параметр | Значение |
|---|---|
| Процессор | Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz |
| Логических ядер | 6 |
| Векторные расширения | sse4_2 avx |
| Память | 11.6 ГБ |
| Виртуализация | kvm |
| ОС | Ubuntu 24.04.4 LTS, ядро 6.8.0-137-generic |
| Python | 3.12.3 |
| onnxruntime | 1.31.0 · providers: AzureExecutionProvider, CPUExecutionProvider |
| numpy | 1.26.4 |
| T-one (пакет tone) | 0.1.0 |
| pyctcdecode / kenlm | 0.5.0 / 0.3.0 |

Отсутствие AVX2 и FMA означает, что onnxruntime работает по медленному пути: цифры такой машины — нижняя граница, а не ориентир для современного CPU.
