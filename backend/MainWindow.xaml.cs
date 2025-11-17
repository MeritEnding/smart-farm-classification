using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
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


// SixLabors.ImageSharp.Rectangle을 사용하기 위해 (int Box)
using Rectangle = SixLabors.ImageSharp.Rectangle;
// [수정] SixLabors.ImageSharp.Point의 모호성을 해결하기 위해 별칭 사용
using SixPoint = SixLabors.ImageSharp.Point;

namespace MangoClassifierWPF
{
    // 분류 모델 결과
    public class PredictionScore
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
    }

    // 탐지 모델 결과 (망고 탐지, 결함 탐지 공용)
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
        private InferenceSession? _defectSession;         // (defect_detection.onnx - 망고 결함)

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


        // --- 결함 탐지 모델 (defect_detection.onnx) 설정 ---
        private readonly string[] _defectClassNames = new string[]
        {
        "brown-spot",         // data.yaml의 0번째 이름
            "black-spot",         // data.yaml의 1번째 이름
            "scab"                // data.yaml의 2번째 이름
        };
        private const int DefectInputSize = 640; // Colab 학습 시 640 사용


        public MainWindow()
        {
            InitializeComponent();
            LoadModelsAsync();

            // (참고) 환경/날씨 텍스트 (임시)
            FarmEnvTextBlock.Text = "온도: 28°C\n습도: 75%";
            WeatherTextBlock.Text = "맑음, 32°C\n바람: 3m/s";
            SeasonInfoTextBlock.Text = "수확기 (7월)\n품종: 애플망고";
        }

        /// <summary>
        /// 3개 모델을 비동기식으로 로드 (UI 차단 방지)
        /// </summary>
        private async void LoadModelsAsync()
        {
            DetectionResultTextBlock.Text = "모델 로드 중...";
            DetectedSizeTextBlock.Text = "...";
            RipenessResultTextBlock.Text = "모델 로드 중...";
            ConfidenceTextBlock.Text = "...";
            DefectResultsTextBlock.Text = "...";
            FinalDecisionTextBlock.Text = "...";

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

                    // 3. 결함 탐지 모델 (defect_detection.onnx) 로드
                    string defectModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "defect_detection.onnx");
                    if (!File.Exists(defectModelPath))
                    {
                        Dispatcher.Invoke(() => MessageBox.Show($"결함 탐지 모델 파일을 찾을 수 없습니다: {defectModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error));
                        return;
                    }
                    _defectSession = new InferenceSession(defectModelPath, sessionOptions);
                });

                // 3개 모델이 모두 로드되었는지 확인
                if (_classificationSession != null && _detectionSession != null && _defectSession != null)
                {
                    DetectionResultTextBlock.Text = "모델 3개 로드 성공.";
                    DetectedSizeTextBlock.Text = "...";
                    RipenessResultTextBlock.Text = "이미지를 선택하세요.";
                    DefectResultsTextBlock.Text = "대기 중";
                    FinalDecisionTextBlock.Text = "대기 중";
                }
                else
                {
                    DetectionResultTextBlock.Text = "모델 로드 실패.";
                    DetectedSizeTextBlock.Text = "---";
                    DefectResultsTextBlock.Text = "---";
                    FinalDecisionTextBlock.Text = "오류";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"모델 로드 중 심각한 오류 발생: {ex.Message}", "모델 로드 실패", MessageBoxButton.OK, MessageBoxImage.Error);
                DetectionResultTextBlock.Text = "모델 로드 실패.";
                DetectedSizeTextBlock.Text = "---";
                DefectResultsTextBlock.Text = "---";
                FinalDecisionTextBlock.Text = "오류";
            }
        }

        /// <summary>
        /// 이미지 버튼 클릭 시 UI 초기화 및 분석 파이프라인 시작
        /// </summary>
        private async void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            // 3개 모델 확인

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
                    DetectionCanvas.Children.Clear();

                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.EndInit();
                    bitmap.Freeze(); // 스레드간 공유를 위해 Freeze

                    SourceImage.Source = bitmap;

                    // [핵심] Viewbox 내부 Grid 크기를 원본 이미지 크기로 설정
                    PreviewGrid.Width = bitmap.PixelWidth;
                    PreviewGrid.Height = bitmap.PixelHeight;

                    DetectionResultTextBlock.Text = "탐지 중...";
                    DetectedSizeTextBlock.Text = "...";
                    RipenessResultTextBlock.Text = "대기 중...";
                    ConfidenceTextBlock.Text = "...";
                    FullResultsListView.ItemsSource = null;
                    DefectResultsTextBlock.Text = "결함 탐지 중...";
                    FinalDecisionTextBlock.Text = "판단 중...";

                    // 비동기로 전체 파이프라인 실행
                    await RunFullPipelineAsync(imagePath);
                }
                catch (Exception ex)
                {
                    DetectionResultTextBlock.Text = "파이프라인 오류";
                    DetectedSizeTextBlock.Text = "---";
                    RipenessResultTextBlock.Text = "---";
                    ConfidenceTextBlock.Text = "---";
                    DefectResultsTextBlock.Text = "오류";
                    FinalDecisionTextBlock.Text = "오류";
                    MessageBox.Show($"처리 중 오류 발생: {ex.Message}\n\n[코딩 파트너 조언]\n'data.yaml'의 'names:' 목록(3개)이 C# 코드의 '_defectClassNames' 배열과 정확히 일치하는지 확인해주세요.", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }


        /// <summary>
        /// 전체 파이프라인 (최종 결론 로직 포함)
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
                    // 탐지 실패 시, 전체 이미지를 Box로 사용
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
                    // 탐지 성공 시
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
                // englishPredictedClass를 추가로 받음 (로직용)
                var (koreanPredictedClass, englishPredictedClass, confidence, allScores)
                    = await RunClassificationAsync(originalImage, cropBox);

                // --- 단계 3B: 결함 탐지 (defect_detection.onnx) ---
                var defectResults = await RunDefectDetectionAsync(originalImage, cropBox);

                // --- 단계 3C: 최종 결론 도출 ---
                var (decision, color, decisionColor) = GetFinalDecision(englishPredictedClass, defectResults, topDetection.Box);


                // --- 단계 4: UI 업데이트 ---
                string estimatedWeight = EstimateWeightCategory(topDetection.Box);

                DetectionResultTextBlock.Text = detectionText;
                DetectedSizeTextBlock.Text = estimatedWeight;
                RipenessResultTextBlock.Text = $"{koreanPredictedClass}"; // 한글 이름 표시
                ConfidenceTextBlock.Text = $"{confidence * 100:F2} %";
                FullResultsListView.ItemsSource = allScores.OrderByDescending(s => s.Confidence);

                // 최종 결론 UI 업데이트
                FinalDecisionTextBlock.Text = decision;
                FinalDecisionTextBlock.Foreground = color; // 텍스트 색상 (보통 White)

                // Border의 배경색 변경 (Parent 캐스팅)
                if (FinalDecisionTextBlock.Parent is Border decisionBorder)
                {
                    decisionBorder.Background = decisionColor;
                }


                // 결함 탐지 결과 UI 업데이트
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
                    DrawBox(topDetection.Box, Brushes.OrangeRed, 3);
                }

                foreach (var defect in defectResults)
                {
                    DrawBox(defect.Box, Brushes.Yellow, 2);
                }
            }
        }


        // -----------------------------------------------------------------
        // [ 핵심 로직: 최종 판매 결정 ]
        // -----------------------------------------------------------------
        /// <summary>
        /// 제공된 매트릭스를 기반으로 최종 판매 결정을 내립니다.
        /// </summary>
        /// <returns>(결정 텍스트, 텍스트 색상, 배경 색상)</returns>
        private (string Decision, Brush TextColor, Brush BackgroundColor) GetFinalDecision(string englishRipeness, List<DetectionResult> defects, Rectangle mangoBox)
        {
            // 'Color.FromRgb'가 모호하므로 'System.Windows.Media.Color.FromRgb'로 명시합니다.
            Brush PASS_COLOR = new SolidColorBrush(System.Windows.Media.Color.FromRgb(0x2E, 0xCC, 0x71));
            Brush REJECT_COLOR = Brushes.DarkRed;
            Brush CONDITIONAL_COLOR = Brushes.DarkOrange;
            Brush HOLD_COLOR = Brushes.DarkSlateGray;
            Brush TEXT_COLOR = Brushes.White;

            // --- 1. 결함 면적 비율 (Defect Ratio) 계산 ---
            double mangoArea = (double)mangoBox.Width * mangoBox.Height;
            if (mangoArea == 0) return ("폐기 (망고 크기 오류)", TEXT_COLOR, REJECT_COLOR);

            double totalDefectArea = 0;
            foreach (var defect in defects)
            {
                var effectiveDefectBox = defect.Box;
                effectiveDefectBox.Intersect(mangoBox);
                totalDefectArea += (double)effectiveDefectBox.Width * effectiveDefectBox.Height;
            }
            double defectRatio = (totalDefectArea / mangoArea);

            // --- 2. 결함 종류 확인 ---
            bool hasScab = defects.Any(d => d.ClassName == "scab");
            bool hasBrownSpot = defects.Any(d => d.ClassName == "brown-spot");
            bool hasBlackSpot = defects.Any(d => d.ClassName == "black-spot");
            bool hasOtherDefects = defects.Any(d => d.ClassName != "black-spot");

            // --- 3. 폐기 기준 (Discard Rules) - 최우선 적용 ---
            if (englishRipeness == "overripe")
                return ("폐기 (과숙)", TEXT_COLOR, REJECT_COLOR);
            if (englishRipeness == "un-healthy")
                return ("폐기 (비정상/병함)", TEXT_COLOR, REJECT_COLOR);
            if (defectRatio > 0.10)
                return ($"폐기 (결함 면적 {defectRatio:P0} > 10%)", TEXT_COLOR, REJECT_COLOR);
            if (hasScab && defectRatio > 0.05)
                return ($"폐기 (Scab 결함 5% 초과)", TEXT_COLOR, REJECT_COLOR);
            if (hasBrownSpot && defectRatio > 0.05)
                return ($"폐기 (Brown Spot 5% 초과)", TEXT_COLOR, REJECT_COLOR);

            // --- 4. 통과 기준 (Pass Rules) ---
            bool passRipeness = (englishRipeness == "half-riping-stage" || englishRipeness == "ripe");
            bool passDefectRatio = (defectRatio <= 0.05);
            bool passDefectType = !hasOtherDefects; // black-spot만 있거나, 아예 없거나

            if (passRipeness && passDefectRatio && passDefectType)
                return ("정상 판매 가능", TEXT_COLOR, PASS_COLOR);

            // --- 5. 조건부 통과 기준 (Conditional Rules) ---
            bool condRipeness = (englishRipeness == "breaking - stage" || englishRipeness == "ripe");
            bool condDefectRatio = (defectRatio > 0.05 && defectRatio <= 0.10);
            bool condDefectType = !hasScab; // Scab만 없으면 됨

            if (condRipeness && condDefectRatio && condDefectType)
                return ("저가 판매 / 즉시 유통", TEXT_COLOR, CONDITIONAL_COLOR);

            // --- 6. 기타 (규칙 외) ---
            if (englishRipeness == "unripe")
                return ("판단 보류 (미숙)", TEXT_COLOR, HOLD_COLOR);

            // 모든 규칙에 맞지 않는 경우
            return ("판단 보류 (규칙 외)", TEXT_COLOR, HOLD_COLOR);
        }

        // -----------------------------------------------------------------
        // [ 이하 모델 추론 및 헬퍼 함수 ]
        // -----------------------------------------------------------------


        /// <summary>
        /// 단계 3B: 'defect_detection.onnx'를 실행하여 결함을 탐지합니다.
        /// [★ 수정된 함수]
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
                    // [★ 수정] 반환값으로 scale, padX, padY를 받음
                    var (resizedImage, scale, padX, padY) = PreprocessDetectionImage(croppedImage, DefectInputSize);

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
                        int numClasses = _defectClassNames.Length;
                        int numBoxes = output.Dimensions[2];

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

                            // 결함 신뢰도 30% 이상만
                            if (maxClassConf > 0.3)
                            {
                                float x_center = output[0, 0, i];
                                float y_center = output[0, 1, i];
                                float w = output[0, 2, i];
                                float h = output[0, 3, i];

                                // --- [★ 핵심 수정] ---
                                // 640x640 좌표 -> 패딩 제거 -> 스케일 역산
                                float left = (x_center - w / 2 - padX) / scale;
                                float top = (y_center - h / 2 - padY) / scale;
                                float right = (x_center + w / 2 - padX) / scale;
                                float bottom = (y_center + h / 2 - padY) / scale;
                                // --- [★ 여기까지] ---

                                // (이 로직은 정답) (잘린 이미지 좌표) + (원본 이미지 오프셋)
                                detectedObjects.Add(new DetectionResult
                                {
                                    ClassName = _defectClassNames[maxClassId],
                                    Confidence = maxClassConf,
                                    Box = new Rectangle(
                                        (int)left + cropBox.X,    // 원본 이미지 X좌표로 오프셋
                                        (int)top + cropBox.Y,     // 원본 이미지 Y좌표로 오프셋
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
        /// 픽셀 면적을 기반으로 무게 범주를 "추정"합니다.
        /// </summary>
        private string EstimateWeightCategory(Rectangle box)
        {
            long area = box.Width * box.Height;
            const long THRESHOLD_SMALL = 50000;
            const long THRESHOLD_MEDIUM = 100000;
            const long THRESHOLD_LARGE = 150000;

            if (area < THRESHOLD_SMALL) return "소 (150-300g)";
            else if (area < THRESHOLD_MEDIUM) return "중 (350-500g)";
            else if (area < THRESHOLD_LARGE) return "대 (500-650g)";
            else return "특대 (600-750g)";
        }

        /// <summary>
        /// 단계 1: 'detection.onnx' (YOLOv8)를 실행하여 *망고 전체*를 탐지합니다.
        /// [★ 수정된 함수]
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
                    // [★ 수정] 반환값으로 scale, padX, padY를 받음
                    var (resizedImage, scale, padX, padY) = PreprocessDetectionImage(image, DetectionInputSize);

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
                        int numBoxes = output.Dimensions[2];

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

                                // --- [★ 핵심 수정] ---
                                // 640x640 좌표 -> 패딩 제거 -> 스케일 역산
                                float left = (x_center - w / 2 - padX) / scale;
                                float top = (y_center - h / 2 - padY) / scale;
                                float right = (x_center + w / 2 - padX) / scale;
                                float bottom = (y_center + h / 2 - padY) / scale;
                                // --- [★ 여기까지] ---

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
        /// 탐지 모델용 전처리 헬퍼 (Pad/Resize)
        /// [★ 수정된 함수]
        /// </summary>
        /// <returns> (처리된 이미지, 스케일 비율, X패딩, Y패딩) </returns>
        private (Image<Rgb24> ProcessedImage, float Scale, int PadX, int PadY) PreprocessDetectionImage(Image<Rgb24> original, int targetSize)
        {
            // 1. 비율 계산
            var scale = new SizeF((float)targetSize / original.Width, (float)targetSize / original.Height);
            float resizeScale = Math.Min(scale.Width, scale.Height); // 작은 쪽을 기준으로 스케일링
            int newWidth = (int)(original.Width * resizeScale);
            int newHeight = (int)(original.Height * resizeScale);

            // 2. 리사이즈
            var resized = original.Clone(ctx => ctx.Resize(newWidth, newHeight, KnownResamplers.Triangle));

            // 3. 패딩 계산
            int padX = (targetSize - newWidth) / 2;
            int padY = (targetSize - newHeight) / 2;

            // 4. 회색 배경(114)에 리사이즈된 이미지 그리기
            var finalImage = new Image<Rgb24>(targetSize, targetSize, new Rgb24(114, 114, 114));

            // [수정] 모호한 'Point' 대신 'SixPoint' (SixLabors.ImageSharp.Point) 사용
            finalImage.Mutate(ctx => ctx.DrawImage(resized, new SixPoint(padX, padY), 1f));

            resized.Dispose();

            // 5. [★ 수정] 스케일 값과 패딩 값을 모두 반환
            return (finalImage, resizeScale, padX, padY);
        }

        /// <summary>
        /// 단계 3A: 'best.onnx'를 실행하여 익음 정도를 분류합니다.
        /// (로직 변경 없음)
        /// </summary>
        private async Task<(string KoreanTopClass, string EnglishTopClass, float TopConfidence, List<PredictionScore> AllScores)> RunClassificationAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_classificationSession == null)
                throw new InvalidOperationException("분류 세션이 초기화되지 않았습니다.");

            return await Task.Run(() =>
            {
                // 1. 탐지된 영역으로 자르기 + 2. 분류 모델 크기로 리사이즈 (Crop 모드)
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
                        var probabilities = Softmax(output.ToArray()); // Softmax 적용

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
                        string koreanTopClass = _translationMap[englishTopClass];    // (표시용)

                        // 영문/한글 이름 모두 반환
                        return (koreanTopClass, englishTopClass, maxConfidence, allScores);
                    }
                }
            });
        }

        /// <summary>
        /// Softmax 함수 (분류 모델 출력에 적용)
        /// </summary>
        private float[] Softmax(float[] logits)
        {
            var maxLogit = logits.Max();
            var exps = logits.Select(l => (float)Math.Exp(l - maxLogit));
            var sumExps = exps.Sum();
            return exps.Select(e => e / sumExps).ToArray();
        }


        // -----------------------------------------------------------------
        // [ Viewbox 방식에 최적화된 그리기 함수 ]
        // (로직 변경 없음)
        // -----------------------------------------------------------------

        /// <summary>
        /// 캔버스(Canvas)에 *하나의* 바운딩 박스를 그립니다.
        /// (Viewbox 방식이므로 스케일링/오프셋 계산이 필요 없습니다.)
        /// </summary>
        private void DrawBox(Rectangle modelBox, Brush strokeBrush, double strokeThickness)
        {
            var rect = new System.Windows.Shapes.Rectangle
            {
                Stroke = strokeBrush,
                StrokeThickness = strokeThickness,
                Width = modelBox.Width,  // 모델이 반환한 원본 너비
                Height = modelBox.Height // 모델이 반환한 원본 높이
            };

            // Canvas에 X, Y 위치 설정
            Canvas.SetLeft(rect, modelBox.X); // 모델이 반환한 원본 X
            Canvas.SetTop(rect, modelBox.Y);  // 모델이 반환한 원본 Y

            DetectionCanvas.Children.Add(rect);
        }
    }
}
