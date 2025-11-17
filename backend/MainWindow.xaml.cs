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
// SixLabors.ImageSharp.Point의 모호성을 해결하기 위해 별칭 사용
using SixPoint = SixLabors.ImageSharp.Point;

namespace MangoClassifierWPF
{
    // [분석 이력 저장을 위한 클래스]
    public class AnalysisHistoryItem
    {
        public BitmapImage Thumbnail { get; set; }
        public BitmapImage FullImageSource { get; set; }
        public double OriginalImageWidth { get; set; }
        public double OriginalImageHeight { get; set; }
        public List<DetectionResult> MangoDetections { get; set; }
        public List<DetectionResult> DefectDetections { get; set; }
        public string FileName { get; set; }
        public string DetectionResultText { get; set; }
        public string DetectedSizeText { get; set; }
        public string RipenessResultText { get; set; }
        public string ConfidenceText { get; set; }
        public string FinalDecisionText { get; set; }
        public Brush FinalDecisionBackground { get; set; }
        public Brush FinalDecisionBrush { get; set; }
        public IEnumerable<PredictionScore> AllRipenessScores { get; set; }
        public string DefectListText { get; set; }
        public Brush DefectListForeground { get; set; }
    }

    // [분류 모델 결과]
    public class PredictionScore
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
    }

    // [탐지 모델 결과]
    public class DetectionResult
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
        public Rectangle Box { get; set; }
    }

    public partial class MainWindow : Window
    {
        private InferenceSession? _classificationSession;
        private InferenceSession? _detectionSession;
        private InferenceSession? _defectSession;

        private List<AnalysisHistoryItem> _analysisHistory = new List<AnalysisHistoryItem>();
        private bool _isHistoryLoading = false;

        // --- 분류 모델 (best.onnx) 설정 ---
        // (제공해주신 클래스 이름 및 번역 맵으로 적용됨)
        private readonly string[] _classificationClassNames = new string[]
        { "breaking-stage", "half-ripe-stage","un-healthy", "ripe", "ripe_with_consumable_disease", "unripe" };
        private readonly Dictionary<string, string> _translationMap = new Dictionary<string, string>
        {
            { "half-ripe-stage", "반숙" },
            { "unripe", "미숙" },
            { "breaking-stage", "중숙" },
            { "ripe", "익음" },
            { "un-healthy", "과숙" }, // 'un-healthy'가 '과숙'으로 매핑됨
            { "ripe_with_consumable_disease", "흠과" },
        };
        private const int ClassificationInputSize = 224;

        // --- 탐지 모델 (detection.onnx - 망고 전체) 설정 ---
        private readonly string[] _detectionClassNames = new string[]
        { "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango" };
        private readonly Dictionary<string, string> _detectionTranslationMap = new Dictionary<string, string>
        { { "Mango", "망고" } };
        private const int DetectionInputSize = 640;

        // --- 결함 탐지 모델 (defect_detection.onnx) 설정 ---
        private readonly string[] _defectClassNames = new string[]
        { "brown-spot", "black-spot", "scab" };
        private readonly Dictionary<string, string> _defectTranslationMap = new Dictionary<string, string>
        {
            { "brown-spot", "갈색 반점" },
            { "black-spot", "검은 반점" },
            { "scab", "더뎅이병" }
        };
        private const int DefectInputSize = 640;


        public MainWindow()
        {
            InitializeComponent();
            LoadModelsAsync();

            FarmEnvTextBlock.Text = "온도: 28°C\n습도: 75%";
            WeatherTextBlock.Text = "맑음, 32°C\n바람: 3m/s";
            SeasonInfoTextBlock.Text = "수확기 (7월)\n품종: 애플망고";
        }

        private async void LoadModelsAsync()
        {
            try
            {
                await Task.Run(() =>
                {
                    var sessionOptions = new SessionOptions();
                    sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;

                    string classificationModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "best.onnx");
                    if (!File.Exists(classificationModelPath)) { Dispatcher.Invoke(() => MessageBox.Show($"분류 모델 파일을 찾을 수 없습니다: {classificationModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error)); return; }
                    _classificationSession = new InferenceSession(classificationModelPath, sessionOptions);

                    string detectionModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "detection.onnx");
                    if (!File.Exists(detectionModelPath)) { Dispatcher.Invoke(() => MessageBox.Show($"탐지 모델 파일을 찾을 수 없습니다: {detectionModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error)); return; }
                    _detectionSession = new InferenceSession(detectionModelPath, sessionOptions);

                    string defectModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "defect_detection.onnx");
                    if (!File.Exists(defectModelPath)) { Dispatcher.Invoke(() => MessageBox.Show($"결함 탐지 모델 파일을 찾을 수 없습니다: {defectModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error)); return; }
                    _defectSession = new InferenceSession(defectModelPath, sessionOptions);
                });

                ResetRightPanelToReady();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"모델 로드 중 심각한 오류 발생: {ex.Message}", "모델 로드 실패", MessageBoxButton.OK, MessageBoxImage.Error);
                ResetRightPanelToReady();
            }
        }

        private async void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
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
                await ProcessImageAsync(openFileDialog.FileName);
            }
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            ResetToWelcomeState();
        }

        // -----------------------------------------------------------------
        // [드래그 앤 드롭 이벤트 핸들러]
        // -----------------------------------------------------------------

        private void WelcomePanel_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop)) e.Effects = DragDropEffects.Copy;
            else e.Effects = DragDropEffects.None;
            e.Handled = true;
        }

        private void WelcomePanel_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop)) e.Effects = DragDropEffects.Copy;
            else e.Effects = DragDropEffects.None;
            e.Handled = true;
        }

        private async void WelcomePanel_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (files != null && files.Length > 0)
                {
                    string imagePath = files[0];
                    string ext = System.IO.Path.GetExtension(imagePath).ToLower();

                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
                    {
                        if (_classificationSession == null || _detectionSession == null || _defectSession == null)
                        {
                            MessageBox.Show("모델이 아직 로드되지 않았습니다.", "오류", MessageBoxButton.OK, MessageBoxImage.Warning);
                            return;
                        }
                        await ProcessImageAsync(imagePath);
                    }
                    else
                    {
                        MessageBox.Show("지원하지 않는 파일 형식입니다. (jpg, jpeg, png만 가능)", "파일 오류", MessageBoxButton.OK, MessageBoxImage.Warning);
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // [이력 불러오기 이벤트 핸들러]
        // -----------------------------------------------------------------
        private void HistoryListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isHistoryLoading || e.AddedItems.Count == 0 || e.AddedItems[0] is not AnalysisHistoryItem selectedItem)
            {
                return;
            }

            _isHistoryLoading = true;

            try
            {
                WelcomePanel.Visibility = Visibility.Collapsed;
                ImagePreviewPanel.Visibility = Visibility.Visible;

                SourceImage.Source = selectedItem.FullImageSource;
                PreviewGrid.Width = selectedItem.OriginalImageWidth;
                PreviewGrid.Height = selectedItem.OriginalImageHeight;

                DetectionCanvas.Children.Clear();
                if (selectedItem.MangoDetections != null)
                {
                    foreach (var box in selectedItem.MangoDetections)
                        DrawBox(box.Box, Brushes.OrangeRed, 3);
                }
                if (selectedItem.DefectDetections != null)
                {
                    foreach (var box in selectedItem.DefectDetections)
                        DrawBox(box.Box, Brushes.Yellow, 2);
                }

                DetectionResultTextBlock.Text = selectedItem.DetectionResultText;
                DetectedSizeTextBlock.Text = selectedItem.DetectedSizeText;
                RipenessResultTextBlock.Text = selectedItem.RipenessResultText;
                ConfidenceTextBlock.Text = selectedItem.ConfidenceText;

                DetectionResultTextBlock.Foreground = Brushes.Orange;
                RipenessResultTextBlock.Foreground = Brushes.DodgerBlue;

                FinalDecisionTextBlock.Text = selectedItem.FinalDecisionText;
                if (FinalDecisionTextBlock.Parent is Border decisionBorder)
                {
                    decisionBorder.Background = selectedItem.FinalDecisionBackground;
                }

                FullResultsListView.ItemsSource = selectedItem.AllRipenessScores;
                DefectResultsTextBlock.Text = selectedItem.DefectListText;
                DefectResultsTextBlock.Foreground = selectedItem.DefectListForeground;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"이력을 불러오는 중 오류가 발생했습니다: {ex.Message}", "이력 오류", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                _isHistoryLoading = false;
            }
        }

        // -----------------------------------------------------------------
        // [이미지 처리 및 UI 리셋 헬퍼 함수]
        // -----------------------------------------------------------------

        private async Task ProcessImageAsync(string imagePath)
        {
            _isHistoryLoading = true;
            HistoryListView.SelectedIndex = -1;
            _isHistoryLoading = false;

            try
            {
                DetectionCanvas.Children.Clear();

                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();
                bitmap.Freeze();
                SourceImage.Source = bitmap;

                PreviewGrid.Width = bitmap.PixelWidth;
                PreviewGrid.Height = bitmap.PixelHeight;

                WelcomePanel.Visibility = Visibility.Collapsed;
                ImagePreviewPanel.Visibility = Visibility.Visible;

                DetectionResultTextBlock.Text = "탐지 중...";
                DetectedSizeTextBlock.Text = "분석 중...";
                RipenessResultTextBlock.Text = "분류 중...";
                ConfidenceTextBlock.Text = "분석 중...";
                FullResultsListView.ItemsSource = null;
                DefectResultsTextBlock.Text = "결함 탐지 중...";
                FinalDecisionTextBlock.Text = "판단 중...";
                if (FinalDecisionTextBlock.Parent is Border decisionBorder)
                {
                    decisionBorder.Background = Brushes.DarkSlateGray;
                }

                await RunFullPipelineAsync(imagePath, bitmap);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"처리 중 오류 발생: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                ResetToWelcomeState();
            }
        }

        private void ResetToWelcomeState()
        {
            Dispatcher.Invoke(() =>
            {
                WelcomePanel.Visibility = Visibility.Visible;
                ImagePreviewPanel.Visibility = Visibility.Collapsed;
                SourceImage.Source = null;

                DetectionCanvas.Children.Clear();
                PreviewGrid.Width = double.NaN;
                PreviewGrid.Height = double.NaN;

                ResetRightPanelToReady();

                _isHistoryLoading = true;
                HistoryListView.SelectedIndex = -1;
                _isHistoryLoading = false;
            });
        }

        private void ResetRightPanelToReady()
        {
            if (_classificationSession != null && _detectionSession != null && _defectSession != null)
            {
                DetectionResultTextBlock.Text = "준비 완료";
                DetectionResultTextBlock.Foreground = Brushes.LightGreen;
                RipenessResultTextBlock.Text = "이미지 대기 중";
                RipenessResultTextBlock.Foreground = Brushes.LightGray;
            }
            else
            {
                DetectionResultTextBlock.Text = "모델 로드 실패";
                DetectionResultTextBlock.Foreground = Brushes.Tomato;
                RipenessResultTextBlock.Text = "---";
            }

            DetectedSizeTextBlock.Text = "---";
            ConfidenceTextBlock.Text = "---";
            FullResultsListView.ItemsSource = null;
            DefectResultsTextBlock.Text = "대기 중";
            FinalDecisionTextBlock.Text = "대기 중";

            if (FinalDecisionTextBlock.Parent is Border decisionBorder)
            {
                decisionBorder.Background = Brushes.DarkSlateGray;
            }
        }

        // -----------------------------------------------------------------
        // [ 분석 파이프라인 및 이력 저장 ]
        // -----------------------------------------------------------------

        private async Task RunFullPipelineAsync(string imagePath, BitmapImage bitmap)
        {
            DetectionCanvas.Children.Clear();

            DetectionResult topDetection;
            string detectionText;
            bool detectionSucceeded;
            List<DetectionResult> defectResults;
            string defectListText;
            Brush defectListForeground;

            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                var detectionResults = await RunDetectionAsync(imagePath);

                if (detectionResults == null || !detectionResults.Any())
                {
                    detectionText = "망고 (40% 이하)";
                    topDetection = new DetectionResult
                    {
                        ClassName = "전체 이미지",
                        Confidence = 1.0,
                        Box = new Rectangle(0, 0, originalImage.Width, originalImage.Height)
                    };
                    detectionSucceeded = false;
                }
                else
                {
                    topDetection = detectionResults.OrderByDescending(r => r.Confidence).First();
                    string koreanDetectionName = _detectionTranslationMap.GetValueOrDefault(topDetection.ClassName, topDetection.ClassName);
                    detectionText = $"{koreanDetectionName} ({topDetection.Confidence * 100:F1}%)";
                    detectionSucceeded = true;
                }

                var cropBox = topDetection.Box;
                cropBox.Intersect(new Rectangle(0, 0, originalImage.Width, originalImage.Height));

                if (cropBox.Width <= 0 || cropBox.Height <= 0)
                {
                    DetectionResultTextBlock.Text = "탐지 영역 오류";
                    return;
                }

                var (koreanPredictedClass, englishPredictedClass, confidence, allScores) = await RunClassificationAsync(originalImage, cropBox);
                defectResults = await RunDefectDetectionAsync(originalImage, cropBox);
                var (decision, color, decisionColor) = GetFinalDecision(englishPredictedClass, defectResults, topDetection.Box);

                string estimatedWeight = EstimateWeightCategory(topDetection.Box);

                // --- UI 업데이트 ---
                DetectionResultTextBlock.Text = detectionText;
                DetectedSizeTextBlock.Text = estimatedWeight;
                RipenessResultTextBlock.Text = $"{koreanPredictedClass}";
                ConfidenceTextBlock.Text = $"{confidence * 100:F2} %"; // Softmax 백분율
                FullResultsListView.ItemsSource = allScores;

                DetectionResultTextBlock.Foreground = Brushes.Orange;
                RipenessResultTextBlock.Foreground = Brushes.DodgerBlue;

                FinalDecisionTextBlock.Text = decision;
                FinalDecisionTextBlock.Foreground = color;
                if (FinalDecisionTextBlock.Parent is Border decisionBorder)
                {
                    decisionBorder.Background = decisionColor;
                }

                if (defectResults.Any())
                {
                    StringBuilder defectSummary = new StringBuilder();
                    defectSummary.AppendLine($"결함 {defectResults.Count}건 탐지됨:");
                    foreach (var defect in defectResults.OrderByDescending(d => d.Confidence))
                    {
                        string koreanDefectName = _defectTranslationMap.GetValueOrDefault(defect.ClassName, defect.ClassName);
                        defectSummary.AppendLine($"- {koreanDefectName} ({defect.Confidence:P1})");
                    }
                    defectListText = defectSummary.ToString();
                    defectListForeground = Brushes.Tomato;
                }
                else
                {
                    defectListText = "탐지된 결함 없음 (정상)";
                    defectListForeground = Brushes.LightGreen;
                }
                DefectResultsTextBlock.Text = defectListText;
                DefectResultsTextBlock.Foreground = defectListForeground;

                if (detectionSucceeded)
                {
                    DrawBox(topDetection.Box, Brushes.OrangeRed, 3);
                }
                foreach (var defect in defectResults)
                {
                    DrawBox(defect.Box, Brushes.Yellow, 2);
                }

                // --- 분석 완료 후 이력 저장 ---
                var historyItem = new AnalysisHistoryItem
                {
                    Thumbnail = bitmap,
                    FullImageSource = bitmap,
                    OriginalImageWidth = bitmap.PixelWidth,
                    OriginalImageHeight = bitmap.PixelHeight,
                    FileName = System.IO.Path.GetFileName(imagePath),

                    MangoDetections = new List<DetectionResult> { topDetection },
                    DefectDetections = defectResults,

                    DetectionResultText = detectionText,
                    DetectedSizeText = estimatedWeight,
                    RipenessResultText = koreanPredictedClass,
                    ConfidenceText = $"{confidence * 100:F2} %", // Softmax 백분율

                    FinalDecisionText = decision,
                    FinalDecisionBackground = decisionColor,
                    FinalDecisionBrush = (decisionColor == REJECT_COLOR || decisionColor == CONDITIONAL_COLOR) ? Brushes.Tomato : Brushes.LightGreen,

                    AllRipenessScores = allScores,
                    DefectListText = defectListText,
                    DefectListForeground = defectListForeground
                };

                _analysisHistory.Insert(0, historyItem);

                HistoryListView.ItemsSource = null;
                HistoryListView.ItemsSource = _analysisHistory;
            }
        }

        // --- 최종 결론 색상 멤버 변수 ---
        private readonly Brush PASS_COLOR = new SolidColorBrush(System.Windows.Media.Color.FromRgb(0x2E, 0xCC, 0x71));
        private readonly Brush REJECT_COLOR = Brushes.DarkRed;
        private readonly Brush CONDITIONAL_COLOR = Brushes.DarkOrange;
        private readonly Brush HOLD_COLOR = Brushes.DarkSlateGray;
        private readonly Brush TEXT_COLOR = Brushes.White;

        // -----------------------------------------------------------------
        // [ ★ 수정된 함수] 제공된 표의 규칙을 정확히 구현
        // -----------------------------------------------------------------
        private (string Decision, Brush TextColor, Brush BackgroundColor) GetFinalDecision(string englishRipeness, List<DetectionResult> defects, Rectangle mangoBox)
        {
            // 1. 결함 종류 확인
            bool hasScab = defects.Any(d => d.ClassName == "scab");
            bool hasBlackSpot = defects.Any(d => d.ClassName == "black-spot");
            bool hasBrownSpot = defects.Any(d => d.ClassName == "brown-spot");
            bool noDefects = !defects.Any(); // 결함이 하나도 없는지 확인

            // --- 규칙 1: "더뎅이병(scab)"이 있으면 무조건 판매 금지 (최우선 순위) ---
            if (hasScab)
            {
                return ($"판매 금지 ({_defectTranslationMap["scab"]} 검출)", TEXT_COLOR, REJECT_COLOR);
            }

            // --- 규칙 2: "검은 반점(black-spot)"이 있으면 (scab은 없는 상태) 무조건 제한적 ---
            if (hasBlackSpot)
            {
                // (테이블: 미숙, 중숙, 반숙... 흠과 모두 "제한적"으로 동일)
                return ("제한적 (외관 판매 부적합, 가공용)", TEXT_COLOR, CONDITIONAL_COLOR);
            }

            // --- 규칙 3: "갈색 반점(brown-spot)"만 있거나 "결함 없음" ---
            // (scab과 black-spot은 이미 위에서 걸러졌음)
            // 이제 익음 단계(ripeness)에 따라 분기
            switch (englishRipeness)
            {
                case "unripe": // 미숙
                case "breaking-stage": // 중숙
                    if (hasBrownSpot)
                        return ("제한적 (후숙/가공용, 외관 부적합)", TEXT_COLOR, HOLD_COLOR);
                    else // noDefects
                        return ("제한적 (후숙용 또는 가공용)", TEXT_COLOR, HOLD_COLOR);

                case "half-ripe-stage": // 반숙
                    if (hasBrownSpot)
                        return ("가능 (가공용, 할인 판매)", TEXT_COLOR, PASS_COLOR);
                    else // noDefects
                        return ("가능 (후숙/가공/할인)", TEXT_COLOR, PASS_COLOR);

                case "ripe": // 익음
                    if (hasBrownSpot)
                        return ("가능 (경미 흠집, 일반 판매)", TEXT_COLOR, PASS_COLOR);
                    else // noDefects
                        return ("가능 (일반 판매용)", TEXT_COLOR, PASS_COLOR);

                case "un-healthy": // 과숙 (맵핑 기준)
                    if (hasBrownSpot)
                        return ("가능 (가공용/할인 판매)", TEXT_COLOR, PASS_COLOR);
                    else // noDefects
                        return ("가능 (빠른 판매/가공/할인)", TEXT_COLOR, PASS_COLOR);

                case "ripe_with_consumable_disease": // 흠과
                    // (scab, black-spot은 이미 위에서 걸러짐)
                    if (hasBrownSpot)
                        return ("가능 (가공용/할인 판매)", TEXT_COLOR, PASS_COLOR);
                    else // noDefects
                        return ("가능 (가공용/할인 판매)", TEXT_COLOR, PASS_COLOR);

                default:
                    // 혹시 모를 예외 처리 (예: _classificationClassNames에 오타가 있을 경우)
                    return ("판단 보류 (알 수 없는 익음)", TEXT_COLOR, HOLD_COLOR);
            }
        }

        // -----------------------------------------------------------------
        // [ 이하 모델 추론 및 헬퍼 함수 ]
        // -----------------------------------------------------------------

        private async Task<List<DetectionResult>> RunDefectDetectionAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_defectSession == null) throw new InvalidOperationException("결함 탐지 세션이 초기화되지 않았습니다.");
            return await Task.Run(() =>
            {
                using (var croppedImage = originalImage.Clone(x => x.Crop(cropBox)))
                {
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
                    using (var results = _defectSession.Run(inputs))
                    {
                        var output = results.First(r => r.Name == "output0").AsTensor<float>();
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
                                if (conf > maxClassConf) { maxClassConf = conf; maxClassId = j; }
                            }
                            if (maxClassConf > 0.3)
                            {
                                float x_center = output[0, 0, i], y_center = output[0, 1, i], w = output[0, 2, i], h = output[0, 3, i];
                                float left = (x_center - w / 2 - padX) / scale;
                                float top = (y_center - h / 2 - padY) / scale;
                                float right = (x_center + w / 2 - padX) / scale;
                                float bottom = (y_center + h / 2 - padY) / scale;
                                detectedObjects.Add(new DetectionResult
                                {
                                    ClassName = _defectClassNames[maxClassId],
                                    Confidence = maxClassConf,
                                    Box = new Rectangle((int)left + cropBox.X, (int)top + cropBox.Y, (int)(right - left), (int)(bottom - top))
                                });
                            }
                        }
                        return detectedObjects;
                    }
                }
            });
        }

        private string EstimateWeightCategory(Rectangle box)
        {
            long area = box.Width * box.Height;
            const long THRESHOLD_SMALL = 50000, THRESHOLD_MEDIUM = 100000, THRESHOLD_LARGE = 150000;
            if (area < THRESHOLD_SMALL) return "소 (150-300g)";
            else if (area < THRESHOLD_MEDIUM) return "중 (350-500g)";
            else if (area < THRESHOLD_LARGE) return "대 (500-650g)";
            else return "특대 (600-750g)";
        }

        private async Task<List<DetectionResult>> RunDetectionAsync(string imagePath)
        {
            if (_detectionSession == null) throw new InvalidOperationException("탐지 세션이 초기화되지 않았습니다.");
            return await Task.Run(() =>
            {
                using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
                {
                    var (resizedImage, scale, padX, padY) = PreprocessDetectionImage(image, DetectionInputSize);
                    var tensor = new DenseTensor<float>(new[] { 1, 3, DetectionInputSize, DefectInputSize });
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
                    using (var results = _detectionSession.Run(inputs))
                    {
                        var output = results.First(r => r.Name == "output0").AsTensor<float>();
                        int numClasses = _detectionClassNames.Length, numBoxes = output.Dimensions[2];
                        List<DetectionResult> detectedObjects = new List<DetectionResult>();
                        for (int i = 0; i < numBoxes; i++)
                        {
                            float maxClassConf = 0.0f;
                            int maxClassId = -1;
                            for (int j = 0; j < numClasses; j++)
                            {
                                var conf = output[0, 4 + j, i];
                                if (conf > maxClassConf) { maxClassConf = conf; maxClassId = j; }
                            }
                            if (maxClassConf > 0.5)
                            {
                                float x_center = output[0, 0, i], y_center = output[0, 1, i], w = output[0, 2, i], h = output[0, 3, i];
                                float left = (x_center - w / 2 - padX) / scale;
                                float top = (y_center - h / 2 - padY) / scale;
                                float right = (x_center + w / 2 - padX) / scale;
                                float bottom = (y_center + h / 2 - padY) / scale;
                                detectedObjects.Add(new DetectionResult
                                {
                                    ClassName = _detectionClassNames[maxClassId],
                                    Confidence = maxClassConf,
                                    Box = new Rectangle((int)left, (int)top, (int)(right - left), (int)(bottom - top))
                                });
                            }
                        }
                        return detectedObjects;
                    }
                }
            });
        }

        private (Image<Rgb24> ProcessedImage, float Scale, int PadX, int PadY) PreprocessDetectionImage(Image<Rgb24> original, int targetSize)
        {
            var scale = new SizeF((float)targetSize / original.Width, (float)targetSize / original.Height);
            float resizeScale = Math.Min(scale.Width, scale.Height);
            int newWidth = (int)(original.Width * resizeScale), newHeight = (int)(original.Height * resizeScale);
            var resized = original.Clone(ctx => ctx.Resize(newWidth, newHeight, KnownResamplers.Triangle));
            int padX = (targetSize - newWidth) / 2, padY = (targetSize - newHeight) / 2;
            var finalImage = new Image<Rgb24>(targetSize, targetSize, new Rgb24(114, 114, 114));
            finalImage.Mutate(ctx => ctx.DrawImage(resized, new SixPoint(padX, padY), 1f));
            resized.Dispose();
            return (finalImage, resizeScale, padX, padY);
        }

        // -----------------------------------------------------------------
        // [ ★ 수정] Softmax (백분율) 복원
        // -----------------------------------------------------------------
        private async Task<(string KoreanTopClass, string EnglishTopClass, float TopConfidence, List<PredictionScore> AllScores)> RunClassificationAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_classificationSession == null) throw new InvalidOperationException("분류 세션이 초기화되지 않았습니다.");
            return await Task.Run(() =>
            {
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

                        // [ ★ 수정됨 ] 모델 출력을 이미 확률로 간주합니다.
                        // C#에서 Softmax 함수를 호출하지 않습니다.
                        var probabilities = output.ToArray();

                        var allScores = new List<PredictionScore>();
                        for (int i = 0; i < probabilities.Length; i++)
                        {
                            string englishName = _classificationClassNames[i];
                            string koreanName = _translationMap.GetValueOrDefault(englishName, englishName);

                            allScores.Add(new PredictionScore
                            {
                                ClassName = koreanName,
                                Confidence = probabilities[i] // 모델 원본 확률
                            });
                        }

                        float maxConfidence = probabilities.Max();
                        int maxIndex = Array.IndexOf(probabilities, maxConfidence);

                        string englishTopClass = _classificationClassNames[maxIndex];
                        string koreanTopClass = _translationMap.GetValueOrDefault(englishTopClass, englishTopClass);

                        // [ ★ 수정됨 ] TopConfidence는 이제 모델의 원본 확률입니다.
                        return (koreanTopClass, englishTopClass, maxConfidence, allScores);
                    }
                }
            });
        }

        // -----------------------------------------------------------------
        // [ ★ 추가] Softmax 함수 복원 (백분율 계산용)
        // -----------------------------------------------------------------
        private float[] Softmax(float[] logits)
        {
            var maxLogit = logits.Max();
            var exps = logits.Select(l => (float)Math.Exp(l - maxLogit));
            var sumExps = exps.Sum();
            return exps.Select(e => e / sumExps).ToArray();
        }

        private void DrawBox(Rectangle modelBox, Brush strokeBrush, double strokeThickness)
        {
            var rect = new System.Windows.Shapes.Rectangle
            {
                Stroke = strokeBrush,
                StrokeThickness = strokeThickness,
                Width = modelBox.Width,
                Height = modelBox.Height
            };
            Canvas.SetLeft(rect, modelBox.X);
            Canvas.SetTop(rect, modelBox.Y);
            DetectionCanvas.Children.Add(rect);
        }
    }
}
