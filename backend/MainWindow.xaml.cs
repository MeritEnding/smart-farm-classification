using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text; // [추가됨] StringBuilder
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

// [추가됨] SixLabors.ImageSharp.Rectangle을 사용하기 위해 (int Box)
using Rectangle = SixLabors.ImageSharp.Rectangle;

namespace MangoClassifierWPF
{
    // 분류 모델 결과
    public class PredictionScore
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
    }

    // [수정됨] 탐지 모델 결과 (망고 탐지, 결함 탐지 공용)
    public class DetectionResult
    {
        public string ClassName { get; set; } = ""; // 예: "망고" 또는 "anthracnose"
        public double Confidence { get; set; } // 예: 0.95
        public Rectangle Box { get; set; } // 이미지 내의 위치 (x, y, width, height)
    }

    public partial class MainWindow : Window
    {
        private InferenceSession? _classificationSession; // (best.onnx)
        private InferenceSession? _detectionSession;      // (detection.onnx - 망고 전체)
        private InferenceSession? _defectSession;         // [신규] (defect_detection.onnx - 망고 결함)

        // --- 분류 모델 (best.onnx) 설정 ---
        private readonly string[] _classificationClassNames = new string[]
        { "overripe", "breaking - stage","un-healthy", "ripe", "unripe", "half-riping-stage" };

        private readonly Dictionary<string, string> _translationMap = new Dictionary<string, string>
        {
            { "breaking - stage", "익어가는 중" },
            { "half-riping-stage", "반숙" },
            { "overripe", "과숙 (지나치게 익음)" },
            { "ripe", "익음 (정상)" },
            { "un-healthy", "비정상 (병든 망고)" },
            { "unripe", "안 익음 (미숙)" }
        };
        private const int ClassificationInputSize = 224;

        // --- 탐지 모델 (detection.onnx - 망고 전체) 설정 ---
        private readonly string[] _detectionClassNames = new string[]
        {
            "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango",
            "Mango", "Mango", "Mango"
        };
        private const int DetectionInputSize = 640;


        // --- [신규] 결함 탐지 모델 (defect_detection.onnx) 설정 ---
        // (이전에 입력해주신 3개 클래스 이름이 반영된 상태입니다)
        private readonly string[] _defectClassNames = new string[]
        {
            "brown-spot",          // data.yaml의 0번째 이름
            "black-spot",          // data.yaml의 1번째 이름
            "scab"                 // data.yaml의 2번째 이름
        };
        private const int DefectInputSize = 640; // Colab 학습 시 640 사용


        public MainWindow()
        {
            InitializeComponent();
            LoadModelsAsync();
        }

        /// <summary>
        /// (수정) 3개 모델을 비동기식으로 로드 (UI 차단 방지)
        /// </summary>
        private async void LoadModelsAsync()
        {
            DetectionResultTextBlock.Text = "모델 로드 중...";
            DetectedSizeTextBlock.Text = "...";
            RipenessResultTextBlock.Text = "모델 로드 중...";
            ConfidenceTextBlock.Text = "...";
            DefectResultsTextBlock.Text = "...";
            FinalDecisionTextBlock.Text = "..."; // [신규]

            try
            {
                await Task.Run(() =>
                {
                    var sessionOptions = new SessionOptions();
                    sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;

                    // 1. 분류 모델 (best.onnx) 로드
                    string classificationModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "best.onnx");
                    if (!File.Exists(classificationModelPath))
                    {
                        Dispatcher.Invoke(() => MessageBox.Show($"분류 모델 파일을 찾을 수 없습니다: {classificationModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error));
                        return;
                    }
                    _classificationSession = new InferenceSession(classificationModelPath, sessionOptions);

                    // 2. 탐지 모델 (detection.onnx) 로드
                    string detectionModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "detection.onnx");
                    if (!File.Exists(detectionModelPath))
                    {
                        Dispatcher.Invoke(() => MessageBox.Show($"탐지 모델 파일을 찾을 수 없습니다: {detectionModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error));
                        return;
                    }
                    _detectionSession = new InferenceSession(detectionModelPath, sessionOptions);

                    // 3. [신규] 결함 탐지 모델 (defect_detection.onnx) 로드
                    string defectModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "defect_detection.onnx");
                    if (!File.Exists(defectModelPath))
                    {
                        Dispatcher.Invoke(() => MessageBox.Show($"결함 탐지 모델 파일을 찾을 수 없습니다: {defectModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error));
                        return;
                    }
                    _defectSession = new InferenceSession(defectModelPath, sessionOptions);
                });

                // [수정] 3개 모델이 모두 로드되었는지 확인
                if (_classificationSession != null && _detectionSession != null && _defectSession != null)
                {
                    DetectionResultTextBlock.Text = "모델 3개 로드 성공.";
                    DetectedSizeTextBlock.Text = "...";
                    RipenessResultTextBlock.Text = "이미지를 선택하세요.";
                    DefectResultsTextBlock.Text = "대기 중";
                    FinalDecisionTextBlock.Text = "대기 중"; // [신규]
                }
                else
                {
                    DetectionResultTextBlock.Text = "모델 로드 실패.";
                    DetectedSizeTextBlock.Text = "---";
                    DefectResultsTextBlock.Text = "---";
                    FinalDecisionTextBlock.Text = "---"; // [신규]
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"모델 로드 중 심각한 오류 발생: {ex.Message}", "모델 로드 실패", MessageBoxButton.OK, MessageBoxImage.Error);
                DetectionResultTextBlock.Text = "모델 로드 실패.";
                DetectedSizeTextBlock.Text = "---";
                DefectResultsTextBlock.Text = "---";
                FinalDecisionTextBlock.Text = "오류"; // [신규]
            }
        }

        /// <summary>
        /// (수정) 이미지 버튼 클릭 시 UI 초기화
        /// </summary>
        private async void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            // [수정] 3개 모델 확인
            if (_classificationSession == null || _detectionSession == null || _defectSession == null)
            {
                MessageBox.Show("모델이 아직 로드되지 않았습니다.", "오류", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|모든 파일 (*.*)|*.*",
                Title = "테스트할 이미지 선택"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                string imagePath = openFileDialog.FileName;

                try
                {
                    // --- 0. UI 초기화 ---
                    DetectionCanvas.Children.Clear(); // 캔버스는 여기서 한 번만 초기화

                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.EndInit();
                    SourceImage.Source = bitmap;

                    DetectionResultTextBlock.Text = "탐지 중...";
                    DetectedSizeTextBlock.Text = "...";
                    RipenessResultTextBlock.Text = "대기 중...";
                    ConfidenceTextBlock.Text = "...";
                    FullResultsListView.ItemsSource = null;
                    DefectResultsTextBlock.Text = "결함 탐지 중...";
                    FinalDecisionTextBlock.Text = "판단 중..."; // [신규]

                    await RunFullPipelineAsync(imagePath);
                }
                catch (Exception ex)
                {
                    DetectionResultTextBlock.Text = "파이프라인 오류";
                    DetectedSizeTextBlock.Text = "---";
                    RipenessResultTextBlock.Text = "---";
                    ConfidenceTextBlock.Text = "---";
                    DefectResultsTextBlock.Text = "오류";
                    FinalDecisionTextBlock.Text = "오류"; // [신규]
                    MessageBox.Show($"처리 중 오류 발생: {ex.Message}\n\n[코딩 파트너 조언]\n'data.yaml'의 'names:' 목록(3개)이 C# 코드의 '_defectClassNames' 배열과 정확히 일치하는지 확인해주세요.", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }


        /// <summary>
        /// (수정) 전체 파이프라인 (최종 결론 로직 추가)
        /// </summary>
        private async Task RunFullPipelineAsync(string imagePath)
        {
            // --- 캔버스 초기화 (파이프라인 시작 시 1회) ---
            DetectionCanvas.Children.Clear();

            DetectionResult topDetection;
            string detectionText; // UI 텍스트 임시 저장
            bool detectionSucceeded; // 탐지 성공 여부 플래그

            // --- 단계 1: 망고 객체 탐지 (detection.onnx) ---
            var detectionResults = await RunDetectionAsync(imagePath);

            // --- 단계 2: 이미지 로드 및 Crop Box 결정 ---
            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                if (detectionResults == null || !detectionResults.Any())
                {
                    // [신규] 탐지 실패 시, 전체 이미지를 Box로 사용
                    detectionText = "물체 탐지 실패 (전체 분석)"; // UI 텍스트
                    topDetection = new DetectionResult
                    {
                        ClassName = "전체 이미지", // 내부용
                        Confidence = 1.0,
                        Box = new Rectangle(0, 0, originalImage.Width, originalImage.Height)
                    };
                    detectionSucceeded = false; // 탐지 실패 플래그
                }
                else
                {
                    // [기존] 탐지 성공 시
                    topDetection = detectionResults.OrderByDescending(r => r.Confidence).First();
                    detectionText = $"{topDetection.ClassName} ({topDetection.Confidence * 100:F1}%)"; // UI 텍스트
                    detectionSucceeded = true; // 탐지 성공 플래그
                }

                // --- 단계 3: 이미지 자르기 (Crop) 준비 ---
                var cropBox = topDetection.Box;
                cropBox.Intersect(new Rectangle(0, 0, originalImage.Width, originalImage.Height));

                if (cropBox.Width <= 0 || cropBox.Height <= 0)
                {
                    DetectionResultTextBlock.Text = "탐지 영역 오류";
                    DetectedSizeTextBlock.Text = "---";
                    DefectResultsTextBlock.Text = "---";
                    FinalDecisionTextBlock.Text = "오류";
                    return;
                }

                // --- 단계 3A: 익음 정도 분류 (best.onnx) ---
                // [수정] englishPredictedClass를 추가로 받음 (로직용)
                var (koreanPredictedClass, englishPredictedClass, confidence, allScores)
                    = await RunClassificationAsync(originalImage, cropBox);

                // --- 단계 3B: 결함 탐지 (defect_detection.onnx) ---
                var defectResults = await RunDefectDetectionAsync(originalImage, cropBox);

                // --- [신규] 단계 3C: 최종 결론 도출 ---
                var (decision, color) = GetFinalDecision(englishPredictedClass, defectResults, topDetection.Box);


                // --- 단계 4: UI 업데이트 ---
                string estimatedWeight = EstimateWeightCategory(topDetection.Box);

                DetectionResultTextBlock.Text = detectionText;
                DetectedSizeTextBlock.Text = estimatedWeight;
                RipenessResultTextBlock.Text = $"{koreanPredictedClass}"; // 한글 이름 표시
                ConfidenceTextBlock.Text = $"{confidence * 100:F2} %";
                FullResultsListView.ItemsSource = allScores.OrderByDescending(s => s.Confidence);

                // [신규] 최종 결론 UI 업데이트
                FinalDecisionTextBlock.Text = decision;
                FinalDecisionTextBlock.Foreground = color;

                // [신규] 결함 탐지 결과 UI 업데이트
                if (defectResults.Any())
                {
                    StringBuilder defectSummary = new StringBuilder();
                    defectSummary.AppendLine($"결함 {defectResults.Count}건 탐지됨:");
                    foreach (var defect in defectResults.OrderByDescending(d => d.Confidence))
                    {
                        defectSummary.AppendLine($"- {defect.ClassName} ({defect.Confidence:P1})");
                    }
                    DefectResultsTextBlock.Text = defectSummary.ToString();
                    DefectResultsTextBlock.Foreground = Brushes.Tomato; // 경고색
                }
                else
                {
                    DefectResultsTextBlock.Text = "탐지된 결함 없음 (정상)";
                    DefectResultsTextBlock.Foreground = Brushes.LightGreen; // 정상색
                }

                // --- 단계 5: 바운딩 박스 그리기 ---
                if (detectionSucceeded)
                {
                    DrawBox(topDetection.Box, originalImage.Width, originalImage.Height, Brushes.OrangeRed, 3);
                }

                foreach (var defect in defectResults)
                {
                    DrawBox(defect.Box, originalImage.Width, originalImage.Height, Brushes.Yellow, 2);
                }
            }
        }


        // -----------------------------------------------------------------
        // [ ⬇️ 신규 함수 ⬇️ ]
        // -----------------------------------------------------------------
        /// <summary>
        /// [신규] 제공된 매트릭스를 기반으로 최종 판매 결정을 내립니다.
        /// </summary>
        /// <param name="englishRipeness">분류 모델의 영문 클래스 이름</param>
        /// <param name="defects">탐지된 결함 목록</param>
        /// <param name="mangoBox">망고 전체의 바운딩 박스</param>
        /// <returns>(결정 텍스트, UI용 브러시)</returns>
        private (string Decision, Brush Color) GetFinalDecision(string englishRipeness, List<DetectionResult> defects, Rectangle mangoBox)
        {
            // --- 1. 결함 면적 비율 (Defect Ratio) 계산 ---
            // (주의: 바운딩 박스 기준이므로 100% 정확하지 않은 '추정치'입니다)
            double mangoArea = (double)mangoBox.Width * mangoBox.Height;
            if (mangoArea == 0) return ("폐기 (망고 크기 오류)", Brushes.Tomato); // 0으로 나누기 방지

            double totalDefectArea = 0;
            foreach (var defect in defects)
            {
                // 망고 박스 내에 있는 결함 면적만 계산
                var effectiveDefectBox = defect.Box;
                effectiveDefectBox.Intersect(mangoBox);
                totalDefectArea += (double)effectiveDefectBox.Width * effectiveDefectBox.Height;
            }
            // 망고 면적 대비 결함 면적 비율
            double defectRatio = (totalDefectArea / mangoArea); // 예: 0.10 = 10%

            // --- 2. 결함 종류 확인 ---
            bool hasScab = defects.Any(d => d.ClassName == "scab");
            bool hasBrownSpot = defects.Any(d => d.ClassName == "brown-spot");
            bool hasBlackSpot = defects.Any(d => d.ClassName == "black-spot");
            // "black-spot" 이외의 다른 결함이 있는지 확인
            bool hasOtherDefects = defects.Any(d => d.ClassName != "black-spot");

            // --- 3. 폐기 기준 (Discard Rules) - 최우선 적용 ---
            // "overripe" 또는 "un-healthy"
            if (englishRipeness == "overripe")
                return ("폐기 (과숙)", Brushes.Tomato);
            if (englishRipeness == "un-healthy")
                return ("폐기 (비정상/병함)", Brushes.Tomato);

            // "결함 면적 비율 10% 이상"
            if (defectRatio > 0.10)
                return ($"폐기 (결함 면적 {defectRatio:P0} > 10%)", Brushes.Tomato);

            // "brown-spot이 대면적", "scab이 깊고 넓은 경우"
            // (해석: '깊이'는 알 수 없으므로, 'scab'이나 'brown-spot'이 존재하고, 
            // 면적이 5%를 넘으면 '대면적/넓은 경우'로 가정합니다)
            if (hasScab && defectRatio > 0.05) // 가정: Scab이 5% 초과
                return ($"폐기 (Scab 결함 5% 초과)", Brushes.Tomato);
            if (hasBrownSpot && defectRatio > 0.05) // 가정: Brown Spot이 5% 초과
                return ($"폐기 (Brown Spot 5% 초과)", Brushes.Tomato);


            // --- 4. 통과 기준 (Pass Rules) ---
            // 익음 상태: "half-riping-stage" 또는 "ripe"
            bool passRipeness = (englishRipeness == "half-riping-stage" || englishRipeness == "ripe");
            // 결함 비율: 5% 이하
            bool passDefectRatio = (defectRatio <= 0.05);
            // 결함 종류: "black-spot만 소량 존재" (즉, black-spot 외 다른 결함이 없어야 함)
            bool passDefectType = !hasOtherDefects;

            if (passRipeness && passDefectRatio && passDefectType)
                return ("정상 판매 가능", Brushes.LightGreen);


            // --- 5. 조건부 통과 기준 (Conditional Rules) ---
            // 상태: "breaking-stage" 또는 "ripe"
            bool condRipeness = (englishRipeness == "breaking - stage" || englishRipeness == "ripe");
            // 결함 비율: 5% ~ 10%
            bool condDefectRatio = (defectRatio > 0.05 && defectRatio <= 0.10);
            // 결함 종류: "scab이 아닌 경우"
            bool condDefectType = !hasScab;

            if (condRipeness && condDefectRatio && condDefectType)
                return ("저가 판매 / 즉시 유통", Brushes.Gold);

            // --- 6. 기타 (규칙 외) ---
            if (englishRipeness == "unripe")
                return ("판단 보류 (미숙)", Brushes.LightSkyBlue);

            // 모든 규칙에 맞지 않는 경우 (예: Ripe, 결함 3%, Scab 존재)
            return ("판단 보류 (규칙 외)", Brushes.Gray);
        }

        // -----------------------------------------------------------------
        // [ ⬇️ 이하 함수들은 기존 로직과 (거의) 동일 ⬇️ ]
        // -----------------------------------------------------------------


        /// <summary>
        /// [신규] 단계 3B: 'defect_detection.onnx'를 실행하여 결함을 탐지합니다.
        /// (RunDetectionAsync와 유사하지만, 잘린 이미지를 입력받습니다)
        /// </summary>
        private async Task<List<DetectionResult>> RunDefectDetectionAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_defectSession == null)
                throw new InvalidOperationException("결함 탐지 세션이 초기화되지 않았습니다.");

            return await Task.Run(() =>
            {
                // 1. 탐지된 망고 영역으로 이미지 자르기
                using (var croppedImage = originalImage.Clone(x => x.Crop(cropBox)))
                {
                    // --- 2. 전처리 (Preprocessing) ---
                    // (잘린 이미지를 640x640으로 리사이즈/패딩)
                    var (resizedImage, scale) = PreprocessDetectionImage(croppedImage, DefectInputSize);

                    var tensor = new DenseTensor<float>(new[] { 1, 3, DefectInputSize, DefectInputSize });
                    resizedImage.ProcessPixelRows(accessor =>
                    {
                        for (int y = 0; y < DefectInputSize; y++)
                        {
                            var rowSpan = accessor.GetRowSpan(y);
                            for (int x = 0; x < DefectInputSize; x++)
                            {
                                tensor[0, 0, y, x] = rowSpan[x].R / 255.0f;
                                tensor[0, 1, y, x] = rowSpan[x].G / 255.0f;
                                tensor[0, 2, y, x] = rowSpan[x].B / 255.0f;
                            }
                        }
                    });

                    resizedImage.Dispose();

                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

                    // --- 3. 추론 (Inference) ---
                    using (var results = _defectSession.Run(inputs))
                    {
                        var output = results.First(r => r.Name == "output0").AsTensor<float>();

                        // --- 4. 후처리 (Postprocessing) ---
                        // [수정됨] _defectClassNames.Length는 이제 3이 됩니다.
                        int numClasses = _defectClassNames.Length;
                        int numBoxes = output.Dimensions[2]; // 8400

                        List<DetectionResult> detectedObjects = new List<DetectionResult>();

                        // (YOLOv8 출력 형식 [batch, 4 + numClasses, 8400])
                        // numClasses가 3이므로, 텐서의 두 번째 차원 크기는 7 (4 + 3)이 됩니다.
                        for (int i = 0; i < numBoxes; i++)
                        {
                            float maxClassConf = 0.0f;
                            int maxClassId = -1;

                            // 클래스 스코어 찾기 (박스 좌표[0~3] 다음부터 클래스 스코어)
                            for (int j = 0; j < numClasses; j++) // j는 0, 1, 2
                            {
                                // j=0 -> output[0, 4, i]
                                // j=1 -> output[0, 5, i]
                                // j=2 -> output[0, 6, i]
                                // 이 인덱스들이 텐서의 실제 범위 내에 있게 됩니다.
                                var conf = output[0, 4 + j, i];
                                if (conf > maxClassConf)
                                {
                                    maxClassConf = conf;
                                    maxClassId = j;
                                }
                            }

                            // [조정 가능] 결함 신뢰도 30% 이상만
                            if (maxClassConf > 0.3)
                            {
                                float x_center = output[0, 0, i];
                                float y_center = output[0, 1, i];
                                float w = output[0, 2, i];
                                float h = output[0, 3, i];

                                // 640x640 이미지 기준 좌표 ➔ 원본(잘린) 이미지 기준 좌표로 스케일 복원
                                float left = (x_center - w / 2) / scale.Width;
                                float top = (y_center - h / 2) / scale.Height;
                                float right = (x_center + w / 2) / scale.Width;
                                float bottom = (y_center + h / 2) / scale.Height;

                                // [🚨 중요] 
                                // 박스 좌표계 변환:
                                // (잘린 이미지 기준 좌표) + (잘린 이미지의 원본 내 위치) = (원본 이미지 기준 좌표)
                                detectedObjects.Add(new DetectionResult
                                {
                                    ClassName = _defectClassNames[maxClassId],
                                    Confidence = maxClassConf,
                                    Box = new Rectangle(
                                        (int)left + cropBox.X,    // ⬅️ 원본 이미지 X좌표로 오프셋
                                        (int)top + cropBox.Y,     // ⬅️ 원본 이미지 Y좌표로 오프셋
                                        (int)(right - left),
                                        (int)(bottom - top)
                                        )
                                });
                            }
                        }
                        return detectedObjects;
                    }
                }
            });
        }


        /// <summary>
        /// [신규] 픽셀 면적을 기반으로 무게 범주를 "추정"합니다. (기존과 동일)
        /// (경고: 이 임계값은 카메라 거리가 고정되었다고 가정한 예시입니다)
        /// </summary>
        private string EstimateWeightCategory(Rectangle box)
        {
            // [🚨 이 부분을 반드시 실제 환경에 맞게 조정하세요!]
            long area = box.Width * box.Height;

            // (이 값은 임의의 "예시" 임계값입니다)
            const long THRESHOLD_SMALL = 30000;  // "소"와 "중"의 경계
            const long THRESHOLD_MEDIUM = 50000; // "중"과 "대"의 경계
            const long THRESHOLD_LARGE = 70000;  // "대"와 "특대"의 경계

            // 제공된 자료 기준
            if (area < THRESHOLD_SMALL)
            {
                // 소과종 (150-300g) - (Alphonso)
                return "소 (150-300g)";
            }
            else if (area < THRESHOLD_MEDIUM)
            {
                // 중과종 (350-500g) - (Irwin)
                return "중 (350-500g)";
            }
            else if (area < THRESHOLD_LARGE)
            {
                // 대과종 (500-650g) - (Haden)
                return "대 (500-650g)";
            }
            else
            {
                // 특대과종 (600-750g) - (Kent)
                return "특대 (600-750g)";
            }
        }

        /// <summary>
        /// (수정 없음) 단계 1: 'detection.onnx' (YOLOv8)를 실행하여 *망고 전체*를 탐지합니다.
        /// </summary>
        private async Task<List<DetectionResult>> RunDetectionAsync(string imagePath)
        {
            if (_detectionSession == null)
                throw new InvalidOperationException("탐지 세션이 초기화되지 않았습니다.");

            return await Task.Run(() =>
            {
                using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
                {
                    // --- 1. 전처리 (Preprocessing) ---
                    var (resizedImage, scale) = PreprocessDetectionImage(image, DetectionInputSize);

                    var tensor = new DenseTensor<float>(new[] { 1, 3, DetectionInputSize, DetectionInputSize });
                    resizedImage.ProcessPixelRows(accessor =>
                    {
                        for (int y = 0; y < DetectionInputSize; y++)
                        {
                            var rowSpan = accessor.GetRowSpan(y);
                            for (int x = 0; x < DetectionInputSize; x++)
                            {
                                tensor[0, 0, y, x] = rowSpan[x].R / 255.0f;
                                tensor[0, 1, y, x] = rowSpan[x].G / 255.0f;
                                tensor[0, 2, y, x] = rowSpan[x].B / 255.0f;
                            }
                        }
                    });

                    resizedImage.Dispose();

                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

                    // --- 2. 추론 (Inference) ---
                    using (var results = _detectionSession.Run(inputs))
                    {
                        var output = results.First(r => r.Name == "output0").AsTensor<float>();

                        // --- 3. 후처리 (Postprocessing) ---
                        int numClasses = _detectionClassNames.Length;
                        int numBoxes = output.Dimensions[2]; // 8400

                        List<DetectionResult> detectedObjects = new List<DetectionResult>();

                        for (int i = 0; i < numBoxes; i++)
                        {
                            float maxClassConf = 0.0f;
                            int maxClassId = -1;
                            for (int j = 0; j < numClasses; j++)
                            {
                                var conf = output[0, 4 + j, i];
                                if (conf > maxClassConf)
                                {
                                    maxClassConf = conf;
                                    maxClassId = j;
                                }
                            }

                            if (maxClassConf > 0.5) // (신뢰도 50% 이상만)
                            {
                                float x_center = output[0, 0, i];
                                float y_center = output[0, 1, i];
                                float w = output[0, 2, i];
                                float h = output[0, 3, i];

                                float left = (x_center - w / 2) / scale.Width;
                                float top = (y_center - h / 2) / scale.Height;
                                float right = (x_center + w / 2) / scale.Width;
                                float bottom = (y_center + h / 2) / scale.Height;

                                detectedObjects.Add(new DetectionResult
                                {
                                    ClassName = _detectionClassNames[maxClassId],
                                    Confidence = maxClassConf,
                                    Box = new Rectangle(
                                        (int)left,
                                        (int)top,
                                        (int)(right - left),
                                        (int)(bottom - top)
                                        )
                                });
                            }
                        }
                        return detectedObjects;
                    }
                }
            });
        }

        /// <summary>
        /// (수정됨) 탐지 모델용 전처리 헬퍼 (Pad/Resize) - 입력 크기(targetSize)를 인자로 받도록 수정
        /// </summary>
        private (Image<Rgb24>, SizeF) PreprocessDetectionImage(Image<Rgb24> original, int targetSize)
        {
            // int targetSize = DetectionInputSize; // 640 (기존)
            var scale = new SizeF((float)targetSize / original.Width, (float)targetSize / original.Height);

            float resizeScale = Math.Min(scale.Width, scale.Height);
            int newWidth = (int)(original.Width * resizeScale);
            int newHeight = (int)(original.Height * resizeScale);

            var resized = original.Clone(ctx => ctx.Resize(newWidth, newHeight, KnownResamplers.Triangle));

            int padX = (targetSize - newWidth) / 2;
            int padY = (targetSize - newHeight) / 2;

            var finalImage = new Image<Rgb24>(targetSize, targetSize, new Rgb24(114, 114, 114));

            // [수정] 'Point'가 모호하므로 'SixLabors.ImageSharp.Point'를 명시
            finalImage.Mutate(ctx => ctx.DrawImage(resized,
                new SixLabors.ImageSharp.Point(padX, padY),
                1f));

            resized.Dispose();

            return (finalImage, new SizeF(resizeScale, resizeScale));
        }

        /// <summary>
        /// [수정] 단계 3A: 'best.onnx'를 실행하여 익음 정도를 분류합니다.
        /// (반환 값에 'englishTopClass' 추가)
        /// </summary>
        private async Task<(string KoreanTopClass, string EnglishTopClass, float TopConfidence, List<PredictionScore> AllScores)> RunClassificationAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_classificationSession == null)
                throw new InvalidOperationException("분류 세션이 초기화되지 않았습니다.");

            return await Task.Run(() =>
            {
                // 1. 탐지된 영역으로 자르기 + 2. 분류 모델 크기로 리사이즈
                using (var image = originalImage.Clone(x =>
                   x.Crop(cropBox)
                    .Resize(new ResizeOptions
                    {
                        Size = new SixLabors.ImageSharp.Size(ClassificationInputSize, ClassificationInputSize),
                        Mode = SixLabors.ImageSharp.Processing.ResizeMode.Crop
                    })
                ))
                {
                    var tensor = new DenseTensor<float>(new[] { 1, 3, ClassificationInputSize, ClassificationInputSize });
                    for (int y = 0; y < image.Height; y++)
                    {
                        for (int x = 0; x < image.Width; x++)
                        {
                            var pixel = image[x, y];
                            tensor[0, 0, y, x] = pixel.R / 255.0f;
                            tensor[0, 1, y, x] = pixel.G / 255.0f;
                            tensor[0, 2, y, x] = pixel.B / 255.0f;
                        }
                    }

                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

                    using (var results = _classificationSession.Run(inputs))
                    {
                        var output = results.First().AsTensor<float>();
                        var probabilities = output.ToArray();

                        var allScores = new List<PredictionScore>();
                        for (int i = 0; i < probabilities.Length; i++)
                        {
                            string englishName = _classificationClassNames[i];
                            string koreanName = _translationMap[englishName];

                            allScores.Add(new PredictionScore
                            {
                                ClassName = koreanName,
                                Confidence = probabilities[i]
                            });
                        }

                        float maxConfidence = probabilities.Max();
                        int maxIndex = Array.IndexOf(probabilities, maxConfidence);

                        string englishTopClass = _classificationClassNames[maxIndex]; // (로직용)
                        string koreanTopClass = _translationMap[englishTopClass];  // (표시용)

                        // [수정] 영문/한글 이름 모두 반환
                        return (koreanTopClass, englishTopClass, maxConfidence, allScores);
                    }
                }
            });
        }

        /// <summary>
        /// [수정됨] 캔버스(Canvas)에 *하나의* 바운딩 박스를 그립니다.
        /// (캔버스 초기화 제거, 브러시/두께 인자 추가)
        /// </summary>
        private void DrawBox(Rectangle modelBox, int originalImageWidth, int originalImageHeight, Brush strokeBrush, double strokeThickness)
        {
            // DetectionCanvas.Children.Clear(); // [제거] 

            var imageControl = SourceImage;
            double controlWidth = imageControl.ActualWidth;
            double controlHeight = imageControl.ActualHeight;

            double scale = Math.Min(controlWidth / originalImageWidth, controlHeight / originalImageHeight);
            double scaledWidth = originalImageWidth * scale;
            double scaledHeight = originalImageHeight * scale;
            double offsetX = (controlWidth - scaledWidth) / 2;
            double offsetY = (controlHeight - scaledHeight) / 2;

            var canvasBox = new System.Windows.Rect(
                (modelBox.X * scale) + offsetX,
                (modelBox.Y * scale) + offsetY,
                (modelBox.Width * scale),
                (modelBox.Height * scale)
            );

            var rect = new System.Windows.Shapes.Rectangle
            {
                Stroke = strokeBrush,
                StrokeThickness = strokeThickness,
                Width = canvasBox.Width,
                Height = canvasBox.Height
            };

            Canvas.SetLeft(rect, canvasBox.Left);
            Canvas.SetTop(rect, canvasBox.Top);

            DetectionCanvas.Children.Add(rect);
        }
    }
}
