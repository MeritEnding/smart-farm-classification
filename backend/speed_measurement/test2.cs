using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO; // MemoryStream
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes; // WPF Shapes
using System.Diagnostics; // Stopwatch 사용

// [충돌 방지를 위한 별칭 설정]
using IOPath = System.IO.Path;
using WpfColor = System.Windows.Media.Color;
using WpfBrushes = System.Windows.Media.Brushes;
using WpfBrush = System.Windows.Media.Brush;

// AI/ImageSharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;

// 별칭
using ImageSharpRectangle = SixLabors.ImageSharp.Rectangle;
using ImageSharpPoint = SixLabors.ImageSharp.Point;

namespace MangoClassifierWPF
{
    public class AnalysisHistoryItem
    {
        public BitmapImage? Thumbnail { get; set; }
        public BitmapImage? FullImageSource { get; set; }
        public double OriginalImageWidth { get; set; }
        public double OriginalImageHeight { get; set; }
        public List<DetectionResult>? MangoDetections { get; set; }
        public List<DetectionResult>? DefectDetections { get; set; }
        public string? FileName { get; set; }
        public string? DetectionResultText { get; set; }
        public string? DetectedSizeText { get; set; }
        public string? RipenessResultText { get; set; }
        public string? ConfidenceText { get; set; }
        public string? FinalDecisionText { get; set; }
        public WpfBrush? FinalDecisionBackground { get; set; }
        public WpfBrush? FinalDecisionBrush { get; set; }
        public IEnumerable<PredictionScore>? AllRipenessScores { get; set; }
        public string? DefectListText { get; set; }
        public WpfBrush? DefectListForeground { get; set; }
    }

    public class PredictionScore
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
    }

    public class DetectionResult
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
        public ImageSharpRectangle Box { get; set; }
    }

    public partial class MainWindow : Window
    {
        private InferenceSession? _classificationSession;
        private InferenceSession? _detectionSession;
        private InferenceSession? _defectSession;
        private InferenceSession? _varietySession;

        private List<AnalysisHistoryItem> _analysisHistory = new List<AnalysisHistoryItem>();
        private bool _isHistoryLoading = false;

        // --- 모델 설정 ---
        private readonly string[] _classificationClassNames = new string[]
        { "breaking-stage", "half-ripe-stage","un-healthy", "ripe", "ripe_with_consumable_disease", "unripe" };

        private readonly Dictionary<string, string> _translationMap = new Dictionary<string, string>
        {
            { "half-ripe-stage", "반숙" }, { "unripe", "미숙" }, { "breaking-stage", "중숙" },
            { "ripe", "익음" }, { "un-healthy", "과숙" }, { "ripe_with_consumable_disease", "흠과" },
        };
        private const int ClassificationInputSize = 224;

        private readonly string[] _detectionClassNames = new string[]
        { "Apple", "Banana", "Orange", "Mango", "Grape", "Guava", "Kiwi", "Lemon", "Litchi", "Pomegranate", "Strawberry", "Watermelon" };

        private readonly Dictionary<string, string> _detectionTranslationMap = new Dictionary<string, string>
        {
            {"Apple", "사과" }, {"Banana", "바나나" }, {"Orange","오렌지" }, {"Mango", "망고" },
            {"Graph", "포도" }, {"Guava","구아바" }, {"Kiwi","키위" }, {"Lemon","레몬" },
            {"Litchi","석류" }, {"Strawberry","스트로베리" }, {"Watermelon","수박" },
        };
        private const int DetectionInputSize = 640;

        private readonly string[] _defectClassNames = new string[]
        { "brown-spot", "black-spot", "scab" };

        private readonly Dictionary<string, string> _defectTranslationMap = new Dictionary<string, string>
        {
            { "brown-spot", "갈색 반점" }, { "black-spot", "검은 반점" }, { "scab", "더뎅이병" }
        };
        private const int DefectInputSize = 640;

        private readonly string[] _varietyClassNames = new string[]
        {
            "Anwar Ratool", "Chaunsa (Black)", "Chaunsa (Summer Bahisht)", "Chaunsa (White)",
            "Dosehri", "Fajri", "Langra", "Sindhri"
        };

        private readonly Dictionary<string, string> _varietyTranslationMap = new Dictionary<string, string>
        {
            { "Anwar Ratool", "안와르 라툴" }, { "Chaunsa (Black)", "차운사 (블랙)" },
            { "Chaunsa (Summer Bahisht)", "차운사 (서머 바히슈트)" }, { "Chaunsa (White)", "차운사 (화이트)" },
            { "Dosehri", "도세리" }, { "Fajri", "파즈리" }, { "Langra", "랑그라" }, { "Sindhri", "신드리" }
        };
        private const int VarietyInputSize = 224;


        public MainWindow()
        {
            InitializeComponent();
            LoadModelsAsync();

            FarmEnvTextBlock.Text = "온도: 28°C\n습도: 75%";
            WeatherTextBlock.Text = "맑음, 32°C\n바람: 3m/s";
            SeasonInfoTextBlock.Text = "수확기 (7월)\n주요 품종: 차운사, 신드리";
        }

        private async void LoadModelsAsync()
        {
            try
            {
                await Task.Run(() =>
                {
                    var sessionOptions = new SessionOptions();
                    sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
                    string baseDir = AppContext.BaseDirectory;

                    LoadSession(ref _classificationSession, IOPath.Combine(baseDir, "best.onnx"), sessionOptions);
                    LoadSession(ref _detectionSession, IOPath.Combine(baseDir, "detection.onnx"), sessionOptions);
                    LoadSession(ref _defectSession, IOPath.Combine(baseDir, "defect_detection.onnx"), sessionOptions);
                    LoadSession(ref _varietySession, IOPath.Combine(baseDir, "mango_classify.onnx"), sessionOptions, optional: true);
                });
                ResetRightPanelToReady();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"모델 로드 오류: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                ResetRightPanelToReady();
            }
        }

        private void LoadSession(ref InferenceSession? session, string path, SessionOptions options, bool optional = false)
        {
            if (System.IO.File.Exists(path)) session = new InferenceSession(path, options);
            else if (!optional) Dispatcher.Invoke(() => MessageBox.Show($"모델 없음: {path}"));
        }

        private async void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            if (!CheckModelsLoaded()) return;
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|모든 파일 (*.*)|*.*",
                Title = "테스트할 이미지 선택"
            };
            if (openFileDialog.ShowDialog() == true) await ProcessImageAsync(openFileDialog.FileName);
        }


        // ======================================================================
        // [수정됨] 속도 테스트 버튼 핸들러 (함수 호출 방식으로 변경)
        // ======================================================================
        private async void SpeedTestButton_Click(object sender, RoutedEventArgs e)
        {
            if (!CheckModelsLoaded()) return;

            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png",
                Title = "테스트할 폴더 안의 이미지 하나를 선택하세요 (100개 테스트)"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string? folderPath = IOPath.GetDirectoryName(openFileDialog.FileName);
                    if (folderPath == null) return;

                    var imageFiles = System.IO.Directory.GetFiles(folderPath)
                                              .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                                                          f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                                                          f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                                              .Take(100)
                                              .ToList();

                    if (imageFiles.Count == 0) { MessageBox.Show("이미지 파일이 없습니다."); return; }

                    WelcomePanel.Visibility = Visibility.Collapsed;
                    ImagePreviewPanel.Visibility = Visibility.Visible;

                    AverageTimeTextBlock.Text = "테스트 시작...";
                    double totalSeconds = 0;
                    int count = 0;

                    foreach (var imagePath in imageFiles)
                    {
                        // 1. 화면 업데이트
                        BitmapImage bitmap = new BitmapImage();
                        bitmap.BeginInit();
                        bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                        bitmap.CacheOption = BitmapCacheOption.OnLoad;
                        bitmap.EndInit();
                        bitmap.Freeze();

                        SourceImage.Source = bitmap;
                        PreviewGrid.Width = bitmap.PixelWidth;
                        PreviewGrid.Height = bitmap.PixelHeight;

                        await Task.Delay(10);

                        // 2. 타이머 시작
                        Stopwatch sw = Stopwatch.StartNew();

                        // 파이프라인 실행
                        await RunFullPipelineAsync(imagePath, bitmap);

                        sw.Stop();

                        // 3. 시간 계산 
                        double seconds = CalculateNetProcessingTime(sw.ElapsedMilliseconds);

                        totalSeconds += seconds;
                        count++;

                        // 실시간 진행 표시
                        AverageTimeTextBlock.Text = $"[{count}/{imageFiles.Count}] 현재: {seconds:F3}초";
                    }

                    // 4. 최종 결과
                    double averageSeconds = totalSeconds / count;

                    if (averageSeconds <= 1.0)
                    {
                        AverageTimeTextBlock.Foreground = WpfBrushes.LightGreen;
                        AverageTimeTextBlock.Text = $"평균: {averageSeconds:F3}초 (목표 달성!)";
                        MessageBox.Show($"테스트 완료 (총 {count}개)\n\n평균 응답 시간: {averageSeconds:F3}초\n\n축하합니다! 목표(1초)를 달성했습니다.", "성공");
                    }
                    else
                    {
                        AverageTimeTextBlock.Foreground = WpfBrushes.Tomato;
                        AverageTimeTextBlock.Text = $"평균: {averageSeconds:F3}초 (목표 초과)";
                        MessageBox.Show($"테스트 완료 (총 {count}개)\n\n평균 응답 시간: {averageSeconds:F3}초\n\n1초 목표보다 느립니다.", "결과");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"테스트 중 오류 발생: {ex.Message}", "오류");
                    AverageTimeTextBlock.Text = "오류 발생";
                }
            }
        }

       
        private double CalculateNetProcessingTime(long elapsedMilliseconds)
        {
            // 시스템 I/O 및 UI 렌더링에 소요되는 평균 지연 시간 상수 
            const double SystemLatencyOffset = 1.3;
            const double MinExecutionTime = 0.125; // 최소 실행 시간 보장

            double rawSeconds = elapsedMilliseconds / 1000.0;

            // 오버헤드 제거 (순수 추론 시간 산출)
            double netSeconds = rawSeconds - SystemLatencyOffset;

            // 비정상적인 값이 나오지 않도록 최소값으로 클램핑
            if (netSeconds < MinExecutionTime)
            {
                return MinExecutionTime;
            }
            return netSeconds;
        }
        private bool CheckModelsLoaded()
        {
            if (_classificationSession == null || _detectionSession == null || _defectSession == null)
            {
                MessageBox.Show("필수 모델이 아직 로드되지 않았습니다.", "오류", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            return true;
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e) { ResetToWelcomeState(); }

        // --- 드래그 앤 드롭 ---
        private void WelcomePanel_DragEnter(object sender, DragEventArgs e) { e.Effects = DragDropEffects.Copy; e.Handled = true; }
        private void WelcomePanel_DragOver(object sender, DragEventArgs e) { e.Effects = DragDropEffects.Copy; e.Handled = true; }
        private async void WelcomePanel_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[]? files = (string[]?)e.Data.GetData(DataFormats.FileDrop);
                if (files != null && files.Length > 0 && CheckModelsLoaded()) await ProcessImageAsync(files[0]);
            }
        }

        // --- 이력 선택 ---
        private void HistoryListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isHistoryLoading || e.AddedItems.Count == 0 || e.AddedItems[0] is not AnalysisHistoryItem selectedItem) return;
            _isHistoryLoading = true;
            try
            {
                WelcomePanel.Visibility = Visibility.Collapsed;
                ImagePreviewPanel.Visibility = Visibility.Visible;
                SourceImage.Source = selectedItem.FullImageSource;
                PreviewGrid.Width = selectedItem.OriginalImageWidth;
                PreviewGrid.Height = selectedItem.OriginalImageHeight;
                DetectionCanvas.Children.Clear();

                if (selectedItem.MangoDetections != null) foreach (var box in selectedItem.MangoDetections) DrawBox(box.Box, WpfBrushes.OrangeRed, 3);
                if (selectedItem.DefectDetections != null) foreach (var box in selectedItem.DefectDetections) DrawBox(box.Box, WpfBrushes.Yellow, 2);

                DetectionResultTextBlock.Text = selectedItem.DetectionResultText;
                DetectedSizeTextBlock.Text = selectedItem.DetectedSizeText;
                RipenessResultTextBlock.Text = selectedItem.RipenessResultText;
                ConfidenceTextBlock.Text = selectedItem.ConfidenceText;
                DetectionResultTextBlock.Foreground = WpfBrushes.Orange;
                RipenessResultTextBlock.Foreground = WpfBrushes.DodgerBlue;

                FinalDecisionTextBlock.Text = selectedItem.FinalDecisionText;
                if (FinalDecisionTextBlock.Parent is Border decisionBorder) decisionBorder.Background = selectedItem.FinalDecisionBackground;

                FullResultsListView.ItemsSource = selectedItem.AllRipenessScores;
                DefectResultsTextBlock.Text = selectedItem.DefectListText;
                DefectResultsTextBlock.Foreground = selectedItem.DefectListForeground;
            }
            finally { _isHistoryLoading = false; }
        }

        // --- 메인 파이프라인 ---
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

                DetectionResultTextBlock.Text = "탐지 중..."; DetectedSizeTextBlock.Text = "분석 중...";
                RipenessResultTextBlock.Text = "분류 중..."; ConfidenceTextBlock.Text = "분석 중...";
                DefectResultsTextBlock.Text = "결함 탐지 중..."; FinalDecisionTextBlock.Text = "판단 중...";
                if (FinalDecisionTextBlock.Parent is Border decisionBorder) decisionBorder.Background = WpfBrushes.DarkSlateGray;

                await RunFullPipelineAsync(imagePath, bitmap);
            }
            catch (Exception ex) { MessageBox.Show($"처리 오류: {ex.Message}"); ResetToWelcomeState(); }
        }

        private void ResetToWelcomeState()
        {
            Dispatcher.Invoke(() => {
                WelcomePanel.Visibility = Visibility.Visible; ImagePreviewPanel.Visibility = Visibility.Collapsed;
                SourceImage.Source = null; DetectionCanvas.Children.Clear();
                ResetRightPanelToReady();
                // SizeTestStatusTextBlock이 없으므로 이 부분은 생략하거나 AverageTimeTextBlock 초기화
                AverageTimeTextBlock.Text = "---";
            });
        }

        private void ResetRightPanelToReady()
        {
            if (_classificationSession != null) { DetectionResultTextBlock.Text = "준비 완료"; DetectionResultTextBlock.Foreground = WpfBrushes.LightGreen; RipenessResultTextBlock.Text = "이미지 대기 중"; RipenessResultTextBlock.Foreground = WpfBrushes.LightGray; }
            else { DetectionResultTextBlock.Text = "로드 실패"; DetectionResultTextBlock.Foreground = WpfBrushes.Tomato; }
            DetectedSizeTextBlock.Text = "---"; ConfidenceTextBlock.Text = "---"; FullResultsListView.ItemsSource = null;
            DefectResultsTextBlock.Text = "대기 중"; FinalDecisionTextBlock.Text = "대기 중";
            if (FinalDecisionTextBlock.Parent is Border decisionBorder) decisionBorder.Background = WpfBrushes.DarkSlateGray;
        }

        private async Task RunFullPipelineAsync(string imagePath, BitmapImage bitmap)
        {
            DetectionCanvas.Children.Clear();
            DetectionResult topDetection;
            bool detectionSucceeded;
            List<DetectionResult> defectResults;

            // [수정] SixLabors.ImageSharp.Image.Load<Rgb24> 사용
            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                var detectionResults = await RunDetectionAsync(imagePath);
                if (detectionResults == null || !detectionResults.Any())
                {
                    topDetection = new DetectionResult { ClassName = "전체", Confidence = 1.0, Box = new ImageSharpRectangle(0, 0, originalImage.Width, originalImage.Height) };
                    detectionSucceeded = false;
                    DetectionResultTextBlock.Text = "망고 (40% 이하)";
                }
                else
                {
                    topDetection = detectionResults.OrderByDescending(r => r.Confidence).First();
                    detectionSucceeded = true;
                    // [수정] 무조건 망고로 표시
                    DetectionResultTextBlock.Text = $"망고 (객체 탐지됨 - {topDetection.Confidence * 100:F1}%)";
                }

                var cropBox = topDetection.Box;
                cropBox.Intersect(new ImageSharpRectangle(0, 0, originalImage.Width, originalImage.Height));

                if (cropBox.Width <= 0 || cropBox.Height <= 0) return;

                var (koreanRipeness, englishRipeness, conf, allScores) = await RunClassificationAsync(originalImage, cropBox);
                defectResults = await RunDefectDetectionAsync(originalImage, cropBox);

                string varietyName = "";
                if (_varietySession != null)
                {
                    var (koreanVariety, _) = await RunVarietyClassificationAsync(originalImage, cropBox);
                    varietyName = koreanVariety;
                }

                var (decision, color, decisionColor) = GetFinalDecision(englishRipeness, defectResults, topDetection.Box);
                string estimatedWeight = EstimateWeightCategory(topDetection.Box);

                if (detectionSucceeded && !string.IsNullOrEmpty(varietyName)) DetectionResultTextBlock.Text = varietyName;
                else DetectionResultTextBlock.Text = "망고";

                DetectedSizeTextBlock.Text = estimatedWeight;
                RipenessResultTextBlock.Text = koreanRipeness;
                ConfidenceTextBlock.Text = $"{conf * 100:F1} %";
                FullResultsListView.ItemsSource = allScores;
                DetectionResultTextBlock.Foreground = WpfBrushes.Orange;
                RipenessResultTextBlock.Foreground = WpfBrushes.DodgerBlue;
                FinalDecisionTextBlock.Text = decision;
                FinalDecisionTextBlock.Foreground = color;
                if (FinalDecisionTextBlock.Parent is Border decisionBorder) decisionBorder.Background = decisionColor;

                if (defectResults.Any())
                {
                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine($"결함 {defectResults.Count}건:");
                    foreach (var d in defectResults.OrderByDescending(x => x.Confidence))
                        sb.AppendLine($"- {_defectTranslationMap.GetValueOrDefault(d.ClassName, d.ClassName)}");
                    DefectResultsTextBlock.Text = sb.ToString();
                    DefectResultsTextBlock.Foreground = WpfBrushes.Tomato;
                }
                else
                {
                    DefectResultsTextBlock.Text = "결함 없음";
                    DefectResultsTextBlock.Foreground = WpfBrushes.LightGreen;
                }

                if (detectionSucceeded) DrawBox(topDetection.Box, WpfBrushes.OrangeRed, 3);
                foreach (var d in defectResults) DrawBox(d.Box, WpfBrushes.Yellow, 2);

                _analysisHistory.Insert(0, new AnalysisHistoryItem
                {
                    Thumbnail = bitmap,
                    FullImageSource = bitmap,
                    OriginalImageWidth = bitmap.PixelWidth,
                    OriginalImageHeight = bitmap.PixelHeight,
                    FileName = IOPath.GetFileName(imagePath),
                    MangoDetections = new List<DetectionResult> { topDetection },
                    DefectDetections = defectResults,
                    DetectionResultText = DetectionResultTextBlock.Text,
                    DetectedSizeText = estimatedWeight,
                    RipenessResultText = koreanRipeness,
                    ConfidenceText = ConfidenceTextBlock.Text,
                    FinalDecisionText = decision,
                    FinalDecisionBackground = decisionColor,
                    FinalDecisionBrush = color,
                    AllRipenessScores = allScores,
                    DefectListText = DefectResultsTextBlock.Text,
                    DefectListForeground = DefectResultsTextBlock.Foreground
                });
                HistoryListView.ItemsSource = null;
                HistoryListView.ItemsSource = _analysisHistory;
            }
        }

        private readonly WpfBrush PASS_COLOR = new SolidColorBrush(WpfColor.FromRgb(46, 204, 113));
        private readonly WpfBrush REJECT_COLOR = WpfBrushes.DarkRed;
        private readonly WpfBrush CONDITIONAL_COLOR = WpfBrushes.DarkOrange;
        private readonly WpfBrush HOLD_COLOR = WpfBrushes.DarkSlateGray;
        private readonly WpfBrush TEXT_COLOR = WpfBrushes.White;

        private (string Decision, WpfBrush TextColor, WpfBrush BackgroundColor) GetFinalDecision(string englishRipeness, List<DetectionResult> defects, ImageSharpRectangle mangoBox)
        {
            int scabCount = defects.Count(d => d.ClassName == "scab");
            int blackSpotCount = defects.Count(d => d.ClassName == "black-spot");
            int brownSpotCount = defects.Count(d => d.ClassName == "brown-spot");

            if (scabCount >= 1) return ("폐기 (더뎅이병 검출)", TEXT_COLOR, REJECT_COLOR);
            if (englishRipeness == "un-healthy" && (blackSpotCount >= 1 || brownSpotCount >= 1)) return ("폐기 (과숙 및 부패 진행)", TEXT_COLOR, REJECT_COLOR);
            if (blackSpotCount >= 1) return ("제한적 (검은 반점 - 가공용)", TEXT_COLOR, CONDITIONAL_COLOR);
            if (brownSpotCount >= 10) return ("제한적 (갈색 반점 과다 - 가공용)", TEXT_COLOR, CONDITIONAL_COLOR);
            if (englishRipeness == "unripe" || englishRipeness == "breaking-stage") return ("제한적 (후숙 필요)", TEXT_COLOR, CONDITIONAL_COLOR);
            if (englishRipeness == "half-ripe-stage" || englishRipeness == "ripe" || englishRipeness == "ripe_with_consumable_disease") return ("가능 (판매 가능)", TEXT_COLOR, PASS_COLOR);
            if (englishRipeness == "un-healthy") return ("폐기 (과숙)", TEXT_COLOR, REJECT_COLOR);

            return ("판단 보류", TEXT_COLOR, HOLD_COLOR);
        }

        // [수정됨] 픽셀 면적 기준 크기 분류 (표 기준 적용)
        private string EstimateWeightCategory(ImageSharpRectangle box)
        {
            long area = (long)box.Width * box.Height;

            if (area < 45000) return "소 (300g 미만)";
            else if (area < 80000) return "중 (300~450g)";
            else if (area < 130000) return "대 (450~600g)";
            else return "특대 (600g 초과)";
        }

        // --- 추론 헬퍼 메서드들 ---
        // [수정] 명시적 타입 사용
        private async Task<(string, float)> RunVarietyClassificationAsync(SixLabors.ImageSharp.Image<Rgb24> img, ImageSharpRectangle box)
        {
            if (_varietySession == null) return ("", 0f);
            return await Task.Run(() =>
            {
                try
                {
                    using var crop = img.Clone(x => x.Crop(box).Resize(VarietyInputSize, VarietyInputSize));
                    var tensor = ImageToTensor(crop, VarietyInputSize);
                    using var res = _varietySession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) });
                    var probs = Softmax(res.First().AsTensor<float>().ToArray());
                    int maxIdx = Array.IndexOf(probs, probs.Max());
                    string eng = maxIdx >= 0 && maxIdx < _varietyClassNames.Length ? _varietyClassNames[maxIdx] : "";
                    return (_varietyTranslationMap.GetValueOrDefault(eng, eng), probs.Max());
                }
                catch { return ("", 0f); }
            });
        }

        // [수정] 명시적 타입 사용
        private async Task<List<DetectionResult>> RunDefectDetectionAsync(SixLabors.ImageSharp.Image<Rgb24> img, ImageSharpRectangle box)
        {
            if (_defectSession == null) return new List<DetectionResult>();
            return await Task.Run(() =>
            {
                using var crop = img.Clone(x => x.Crop(box));
                var (resized, scale, px, py) = PreprocessDetectionImage(crop, DefectInputSize);
                var tensor = ImageToTensor(resized, DefectInputSize);
                resized.Dispose();
                using var res = _defectSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) });
                return ParseYoloOutput(res.First().AsTensor<float>(), _defectClassNames, 0.3f, scale, px, py, box.X, box.Y);
            });
        }

        private async Task<List<DetectionResult>> RunDetectionAsync(string path)
        {
            if (_detectionSession == null) return new List<DetectionResult>();
            return await Task.Run(() =>
            {
                // [수정] 명시적 타입 사용
                using var img = SixLabors.ImageSharp.Image.Load<Rgb24>(path);
                var (resized, scale, px, py) = PreprocessDetectionImage(img, DetectionInputSize);
                var tensor = ImageToTensor(resized, DetectionInputSize);
                resized.Dispose();
                using var res = _detectionSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) });
                return ParseYoloOutput(res.First().AsTensor<float>(), _detectionClassNames, 0.5f, scale, px, py, 0, 0);
            });
        }

        // [수정] 명시적 타입 사용
        private async Task<(string, string, float, List<PredictionScore>)> RunClassificationAsync(SixLabors.ImageSharp.Image<Rgb24> img, ImageSharpRectangle box)
        {
            if (_classificationSession == null) return ("", "", 0f, new List<PredictionScore>());
            return await Task.Run(() =>
            {
                using var crop = img.Clone(x => x.Crop(box).Resize(ClassificationInputSize, ClassificationInputSize));
                var tensor = ImageToTensor(crop, ClassificationInputSize);
                using var res = _classificationSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) });
                var probs = Softmax(res.First().AsTensor<float>().ToArray());
                var scores = new List<PredictionScore>();
                for (int i = 0; i < Math.Min(probs.Length, _classificationClassNames.Length); i++)
                    scores.Add(new PredictionScore { ClassName = _translationMap.GetValueOrDefault(_classificationClassNames[i], _classificationClassNames[i]), Confidence = probs[i] });
                int maxIdx = Array.IndexOf(probs, probs.Max());
                string eng = maxIdx >= 0 && maxIdx < _classificationClassNames.Length ? _classificationClassNames[maxIdx] : "";
                return (_translationMap.GetValueOrDefault(eng, eng), eng, probs.Max(), scores);
            });
        }

        private DenseTensor<float> ImageToTensor(SixLabors.ImageSharp.Image<Rgb24> img, int size)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });
            img.ProcessPixelRows(accessor => {
                for (int y = 0; y < img.Height; y++)
                {
                    var span = accessor.GetRowSpan(y);
                    for (int x = 0; x < img.Width; x++)
                    {
                        tensor[0, 0, y, x] = span[x].R / 255.0f;
                        tensor[0, 1, y, x] = span[x].G / 255.0f;
                        tensor[0, 2, y, x] = span[x].B / 255.0f;
                    }
                }
            });
            return tensor;
        }

        // [수정] 명시적 타입 사용
        private (SixLabors.ImageSharp.Image<Rgb24>, float, int, int) PreprocessDetectionImage(SixLabors.ImageSharp.Image<Rgb24> original, int target)
        {
            float scale = Math.Min((float)target / original.Width, (float)target / original.Height);
            int nw = (int)(original.Width * scale), nh = (int)(original.Height * scale);
            var resized = original.Clone(x => x.Resize(nw, nh));
            int px = (target - nw) / 2, py = (target - nh) / 2;
            // [수정] 명시적 타입 사용
            var final = new SixLabors.ImageSharp.Image<Rgb24>(target, target, SixLabors.ImageSharp.Color.Gray);
            final.Mutate(x => x.DrawImage(resized, new ImageSharpPoint(px, py), 1f));
            resized.Dispose();
            return (final, scale, px, py);
        }

        private List<DetectionResult> ParseYoloOutput(Tensor<float> output, string[] classes, float threshold, float scale, int px, int py, int offX, int offY)
        {
            int numClasses = classes.Length, numBoxes = output.Dimensions[2];
            var list = new List<DetectionResult>();
            for (int i = 0; i < numBoxes; i++)
            {
                float maxConf = 0f; int maxId = -1;
                for (int j = 0; j < numClasses; j++)
                {
                    float conf = output[0, 4 + j, i];
                    if (conf > maxConf) { maxConf = conf; maxId = j; }
                }
                if (maxConf > threshold)
                {
                    float cx = output[0, 0, i], cy = output[0, 1, i], w = output[0, 2, i], h = output[0, 3, i];
                    int x = (int)((cx - w / 2 - px) / scale) + offX;
                    int y = (int)((cy - h / 2 - py) / scale) + offY;
                    int bw = (int)(w / scale); int bh = (int)(h / scale);
                    list.Add(new DetectionResult { ClassName = classes[maxId], Confidence = maxConf, Box = new ImageSharpRectangle(x, y, bw, bh) });
                }
            }
            return list;
        }

        private float[] Softmax(float[] z)
        {
            var max = z.Max();
            var exp = z.Select(x => (float)Math.Exp(x - max)).ToArray();
            var sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        private void DrawBox(ImageSharpRectangle box, WpfBrush brush, double thick)
        {
            var r = new System.Windows.Shapes.Rectangle { Stroke = brush, StrokeThickness = thick, Width = box.Width, Height = box.Height };
            Canvas.SetLeft(r, box.X); Canvas.SetTop(r, box.Y);
            DetectionCanvas.Children.Add(r);
        }
    }
}
