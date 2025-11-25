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
using System.Diagnostics; // [추가] 시간을 측정하기 위해 필요

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
        private InferenceSession? _classificationSession; // 익음 정도
        private InferenceSession? _detectionSession;      // 객체 탐지
        private InferenceSession? _defectSession;         // 결함 탐지
        private InferenceSession? _varietySession;        // 품종 분류

        private List<AnalysisHistoryItem> _analysisHistory = new List<AnalysisHistoryItem>();
        private bool _isHistoryLoading = false;

        // --- 익음 분류 모델 (best.onnx) 설정 ---
        private readonly string[] _classificationClassNames = new string[]
        { "breaking-stage", "half-ripe-stage","un-healthy", "ripe", "ripe_with_consumable_disease", "unripe" };

        private readonly Dictionary<string, string> _translationMap = new Dictionary<string, string>
        {
            { "half-ripe-stage", "반숙" },
            { "unripe", "미숙" },
            { "breaking-stage", "중숙" },
            { "ripe", "익음" },
            { "un-healthy", "과숙" },
            { "ripe_with_consumable_disease", "흠과" },
        };
        private const int ClassificationInputSize = 224;

        // --- 탐지 모델 (detection.onnx) 설정 ---
        private readonly string[] _detectionClassNames = new string[]
        { "Apple", "Banana", "Orange", "Mango", "Grape", "Guava", "Kiwi", "Lemon", "Litchi", "Pomegranate", "Strawberry", "Watermelon" };

        private readonly Dictionary<string, string> _detectionTranslationMap = new Dictionary<string, string>
        {
            {"Apple", "사과" }, {"Banana", "바나나" }, {"Orange","오렌지" }, {"Mango", "망고" },
            {"Graph", "포도" }, {"Guava","구아바" }, {"Kiwi","키위" }, {"Lemon","레몬" },
            {"Litchi","석류" }, {"Strawberry","스트로베리" }, {"Watermelon","수박" },
        };
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

        // ================================================================================
        // [★수정됨] 품종 분류 모델 (mango_classify.onnx) 설정
        // 등급(Class_I, II, Extra)을 제외하고 남은 8개 품종
        // ================================================================================
        private readonly string[] _varietyClassNames = new string[]
        {
            "Anwar Ratool",
            "Chaunsa (Black)",
            "Chaunsa (Summer Bahisht)",
            "Chaunsa (White)",
            "Dosehri",
            "Fajri",
            "Langra",
            "Sindhri"
        };

        private readonly Dictionary<string, string> _varietyTranslationMap = new Dictionary<string, string>
        {
            { "Anwar Ratool", "안와르 라툴" },
            { "Chaunsa (Black)", "차운사 (블랙)" },
            { "Chaunsa (Summer Bahisht)", "차운사 (서머 바히슈트)" },
            { "Chaunsa (White)", "차운사 (화이트)" },
            { "Dosehri", "도세리" },
            { "Fajri", "파즈리" },
            { "Langra", "랑그라" },
            { "Sindhri", "신드리" }
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

                    // 1. 익음 분류
                    string classificationModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "best.onnx");
                    if (!File.Exists(classificationModelPath)) { Dispatcher.Invoke(() => MessageBox.Show($"분류 모델 파일을 찾을 수 없습니다: {classificationModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error)); return; }
                    _classificationSession = new InferenceSession(classificationModelPath, sessionOptions);

                    // 2. 객체 탐지
                    string detectionModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "detection.onnx");
                    if (!File.Exists(detectionModelPath)) { Dispatcher.Invoke(() => MessageBox.Show($"탐지 모델 파일을 찾을 수 없습니다: {detectionModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error)); return; }
                    _detectionSession = new InferenceSession(detectionModelPath, sessionOptions);

                    // 3. 결함 탐지
                    string defectModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "defect_detection.onnx");
                    if (!File.Exists(defectModelPath)) { Dispatcher.Invoke(() => MessageBox.Show($"결함 탐지 모델 파일을 찾을 수 없습니다: {defectModelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error)); return; }
                    _defectSession = new InferenceSession(defectModelPath, sessionOptions);

                    // 4. 품종 분류
                    string varietyModelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "mango_classify.onnx");
                    if (File.Exists(varietyModelPath))
                    {
                        _varietySession = new InferenceSession(varietyModelPath, sessionOptions);
                    }
                    else
                    {
                        System.Diagnostics.Debug.WriteLine("품종 모델(mango_classify.onnx)을 찾을 수 없습니다.");
                    }
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
                MessageBox.Show("필수 모델이 아직 로드되지 않았습니다.", "오류", MessageBoxButton.OK, MessageBoxImage.Warning);
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

        // [추가] 속도 테스트 버튼 핸들러
        // [수정됨] 속도 테스트 버튼 핸들러 (이미지 갱신 + 초 단위 표시)
        private async void SpeedTestButton_Click(object sender, RoutedEventArgs e)
        {
            if (_classificationSession == null || _detectionSession == null || _defectSession == null)
            {
                MessageBox.Show("필수 모델이 아직 로드되지 않았습니다.", "오류", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png",
                Title = "테스트할 폴더 안의 이미지 하나를 선택하세요 (해당 폴더의 이미지 100개를 테스트합니다)"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string selectedFile = openFileDialog.FileName;
                    string folderPath = System.IO.Path.GetDirectoryName(selectedFile);

                    // 폴더 내 이미지 파일 가져오기 (최대 100개)
                    var imageFiles = Directory.GetFiles(folderPath)
                                              .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                                                          f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                                                          f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                                              .Take(100)
                                              .ToList();

                    if (imageFiles.Count == 0)
                    {
                        MessageBox.Show("폴더에 이미지 파일이 없습니다.", "알림");
                        return;
                    }

                    // UI 준비 (이미지 패널 보이기)
                    WelcomePanel.Visibility = Visibility.Collapsed;
                    ImagePreviewPanel.Visibility = Visibility.Visible;

                    AverageTimeTextBlock.Text = "테스트 시작...";
                    double totalSeconds = 0;
                    int count = 0;

                    foreach (var imagePath in imageFiles)
                    {
                        // 1. 화면에 이미지 표시 (사용자가 바뀌는 것을 볼 수 있게 함)
                        BitmapImage bitmap = new BitmapImage();
                        bitmap.BeginInit();
                        bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                        bitmap.CacheOption = BitmapCacheOption.OnLoad;
                        bitmap.EndInit();
                        bitmap.Freeze();

                        SourceImage.Source = bitmap;
                        PreviewGrid.Width = bitmap.PixelWidth;
                        PreviewGrid.Height = bitmap.PixelHeight;

                        // ★ 중요: UI가 이미지를 그릴 수 있도록 아주 잠깐 대기 (AI 측정 시간에는 미포함)
                        await Task.Delay(10);

                        // 2. 타이머 시작 (순수 AI 분석 및 결과 표출 시간 측정)
                        Stopwatch sw = Stopwatch.StartNew();

                        // 기존 파이프라인 실행 (오른쪽 결과 패널 업데이트됨)
                        await RunFullPipelineAsync(imagePath, bitmap);

                        sw.Stop();

                        // 3. 시간 계산 (초 단위)
                        double seconds = sw.ElapsedMilliseconds / 1000.0;
                        totalSeconds += seconds;
                        count++;

                        // 실시간 진행 상황 표시 (현재 이미지의 소요 시간)
                        AverageTimeTextBlock.Text = $"[{count}/{imageFiles.Count}] 현재: {seconds:F3}초";

                        // 너무 빠르면 눈으로 확인이 안되므로, 100번 테스트 중 확인을 위해 약간의 딜레이가 필요하면 주석 해제
                        // await Task.Delay(50); 
                    }

                    // 4. 최종 결과 계산
                    double averageSeconds = totalSeconds / count;

                    // 목표 달성 여부에 따른 색상 변경 (1초 이내 목표)
                    if (averageSeconds <= 1.0)
                    {
                        AverageTimeTextBlock.Foreground = Brushes.LightGreen; // 목표 달성!
                        AverageTimeTextBlock.Text = $"평균: {averageSeconds:F3}초 (목표 달성!)";
                        MessageBox.Show($"테스트 완료 (총 {count}개)\n\n평균 응답 시간: {averageSeconds:F3}초\n\n축하합니다! 목표(1초)를 달성했습니다.", "성공");
                    }
                    else
                    {
                        AverageTimeTextBlock.Foreground = Brushes.Tomato; // 목표 실패
                        AverageTimeTextBlock.Text = $"평균: {averageSeconds:F3}초 (목표 초과)";
                        MessageBox.Show($"테스트 완료 (총 {count}개)\n\n평균 응답 시간: {averageSeconds:F3}초\n\n1초 목표보다 느립니다. 최적화가 필요할 수 있습니다.", "결과");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"테스트 중 오류 발생: {ex.Message}", "오류");
                    AverageTimeTextBlock.Text = "오류 발생";
                }
            }
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            ResetToWelcomeState();
        }

        // -----------------------------------------------------------------
        // [드래그 앤 드롭 이벤트]
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
                        MessageBox.Show("지원하지 않는 파일 형식입니다.", "파일 오류", MessageBoxButton.OK, MessageBoxImage.Warning);
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // [이력 불러오기]
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
                MessageBox.Show($"이력 로드 중 오류: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                _isHistoryLoading = false;
            }
        }

        // -----------------------------------------------------------------
        // [이미지 처리 메인 로직]
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

                // [추가] 리셋 시 텍스트 초기화
                AverageTimeTextBlock.Text = "---";
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
            if (FinalDecisionTextBlock.Parent is Border decisionBorder) decisionBorder.Background = Brushes.DarkSlateGray;
        }

        // -----------------------------------------------------------------
        // [ 분석 파이프라인 ]
        // -----------------------------------------------------------------
        private async Task RunFullPipelineAsync(string imagePath, BitmapImage bitmap)
        {
            // 주의: 속도 테스트 시 UI 업데이트가 잦으면 성능 저하가 올 수 있으나,
            // 100회 정도는 사용자 피드백을 위해 그대로 UI를 업데이트하도록 두었습니다.
            DetectionCanvas.Children.Clear();

            DetectionResult topDetection;
            string detectionText;
            bool detectionSucceeded;
            List<DetectionResult> defectResults;
            string defectListText;
            Brush defectListForeground;

            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                // 1. 객체 탐지
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

                // 2. 익음 정도 분석 (안전장치 적용)
                var (koreanPredictedClass, englishPredictedClass, confidence, allScores) = await RunClassificationAsync(originalImage, cropBox);

                // 3. 결함 탐지
                defectResults = await RunDefectDetectionAsync(originalImage, cropBox);

                // 4. 품종 분석 (안전장치 적용)
                string varietyName = "";
                if (_varietySession != null)
                {
                    var (koreanVariety, varietyConf) = await RunVarietyClassificationAsync(originalImage, cropBox);
                    varietyName = koreanVariety;
                }

                // 5. 최종 판정
                var (decision, color, decisionColor) = GetFinalDecision(englishPredictedClass, defectResults, topDetection.Box);
                string estimatedWeight = EstimateWeightCategory(topDetection.Box);

                // --- UI 업데이트 ---
                if (detectionSucceeded && !string.IsNullOrEmpty(varietyName))
                {
                    DetectionResultTextBlock.Text = $"{varietyName}";
                }
                else
                {
                    DetectionResultTextBlock.Text = detectionText;
                }

                DetectedSizeTextBlock.Text = estimatedWeight;
                RipenessResultTextBlock.Text = $"{koreanPredictedClass}";
                ConfidenceTextBlock.Text = $"{confidence * 100:F2} %";
                FullResultsListView.ItemsSource = allScores;

                DetectionResultTextBlock.Foreground = Brushes.Orange;
                RipenessResultTextBlock.Foreground = Brushes.DodgerBlue;

                FinalDecisionTextBlock.Text = decision;
                FinalDecisionTextBlock.Foreground = color;
                if (FinalDecisionTextBlock.Parent is Border decisionBorder) decisionBorder.Background = decisionColor;

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

                if (detectionSucceeded) DrawBox(topDetection.Box, Brushes.OrangeRed, 3);
                foreach (var defect in defectResults) DrawBox(defect.Box, Brushes.Yellow, 2);

                // --- 이력 저장 ---
                var historyItem = new AnalysisHistoryItem
                {
                    Thumbnail = bitmap,
                    FullImageSource = bitmap,
                    OriginalImageWidth = bitmap.PixelWidth,
                    OriginalImageHeight = bitmap.PixelHeight,
                    FileName = System.IO.Path.GetFileName(imagePath),
                    MangoDetections = new List<DetectionResult> { topDetection },
                    DefectDetections = defectResults,
                    DetectionResultText = DetectionResultTextBlock.Text,
                    DetectedSizeText = estimatedWeight,
                    RipenessResultText = koreanPredictedClass,
                    ConfidenceText = $"{confidence * 100:F2} %",
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

        private readonly Brush PASS_COLOR = new SolidColorBrush(System.Windows.Media.Color.FromRgb(0x2E, 0xCC, 0x71));
        private readonly Brush REJECT_COLOR = Brushes.DarkRed;
        private readonly Brush CONDITIONAL_COLOR = Brushes.DarkOrange;
        private readonly Brush HOLD_COLOR = Brushes.DarkSlateGray;
        private readonly Brush TEXT_COLOR = Brushes.White;

        private (string Decision, Brush TextColor, Brush BackgroundColor) GetFinalDecision(string englishRipeness, List<DetectionResult> defects, Rectangle mangoBox)
        {
            bool hasScab = defects.Any(d => d.ClassName == "scab");
            bool hasBlackSpot = defects.Any(d => d.ClassName == "black-spot");
            bool hasBrownSpot = defects.Any(d => d.ClassName == "brown-spot");

            if (hasScab) return ($"판매 금지 ({_defectTranslationMap["scab"]} 검출)", TEXT_COLOR, REJECT_COLOR);
            if (englishRipeness == "un-healthy") return ("판매 금지 (과숙 판정)", TEXT_COLOR, REJECT_COLOR);
            if (hasBlackSpot) return ("제한적 (외관 판매 부적합, 가공용)", TEXT_COLOR, CONDITIONAL_COLOR);

            switch (englishRipeness)
            {
                case "unripe":
                case "breaking-stage":
                    return ("제한적 (후숙/가공용)", TEXT_COLOR, HOLD_COLOR);
                case "half-ripe-stage":
                    return ("가능 (후숙/가공/할인)", TEXT_COLOR, PASS_COLOR);
                case "ripe":
                    return ("가능 (일반 판매용)", TEXT_COLOR, PASS_COLOR);
                case "ripe_with_consumable_disease":
                    return ("가능 (가공용/할인 판매)", TEXT_COLOR, PASS_COLOR);
                default:
                    return ("판단 보류", TEXT_COLOR, HOLD_COLOR);
            }
        }

        // -----------------------------------------------------------------
        // [ 모델 추론 함수 (안전장치 포함) ]
        // -----------------------------------------------------------------

        // 1. 품종 분류 (안전장치: IndexOutOfRange 방지)
        private async Task<(string KoreanName, float Confidence)> RunVarietyClassificationAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_varietySession == null) return ("모델 없음", 0f);

            return await Task.Run(() =>
            {
                try
                {
                    using (var image = originalImage.Clone(x => x.Crop(cropBox).Resize(new ResizeOptions { Size = new SixLabors.ImageSharp.Size(VarietyInputSize, VarietyInputSize), Mode = SixLabors.ImageSharp.Processing.ResizeMode.Crop })))
                    {
                        var tensor = new DenseTensor<float>(new[] { 1, 3, VarietyInputSize, VarietyInputSize });
                        image.ProcessPixelRows(accessor =>
                        {
                            for (int y = 0; y < image.Height; y++)
                            {
                                var rowSpan = accessor.GetRowSpan(y);
                                for (int x = 0; x < image.Width; x++)
                                {
                                    tensor[0, 0, y, x] = rowSpan[x].R / 255.0f;
                                    tensor[0, 1, y, x] = rowSpan[x].G / 255.0f;
                                    tensor[0, 2, y, x] = rowSpan[x].B / 255.0f;
                                }
                            }
                        });

                        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };
                        using (var results = _varietySession.Run(inputs))
                        {
                            var output = results.First().AsTensor<float>().ToArray();
                            var probabilities = Softmax(output);
                            float maxConf = probabilities.Max();
                            int maxIndex = Array.IndexOf(probabilities, maxConf);

                            if (maxIndex >= 0 && maxIndex < _varietyClassNames.Length)
                            {
                                string englishName = _varietyClassNames[maxIndex];
                                return (_varietyTranslationMap.GetValueOrDefault(englishName, englishName), maxConf);
                            }
                            else
                            {
                                System.Diagnostics.Debug.WriteLine($"[경고] 품종 인덱스 초과: {maxIndex}");
                                return ("알 수 없음 (모델 불일치)", maxConf);
                            }
                        }
                    }
                }
                catch (Exception ex) { System.Diagnostics.Debug.WriteLine(ex.Message); }
                return ("오류", 0f);
            });
        }

        // 2. 결함 탐지
        private async Task<List<DetectionResult>> RunDefectDetectionAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_defectSession == null) throw new InvalidOperationException("결함 탐지 세션 없음");
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
            if (area < 50000) return "소 (150-300g)";
            else if (area < 100000) return "중 (350-500g)";
            else if (area < 150000) return "대 (500-650g)";
            else return "특대 (600-750g)";
        }

        // 3. 객체 탐지
        private async Task<List<DetectionResult>> RunDetectionAsync(string imagePath)
        {
            if (_detectionSession == null) throw new InvalidOperationException("탐지 세션 없음");
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

        // 4. 익음 분류 (안전장치: Loop Count 제한)
        private async Task<(string KoreanTopClass, string EnglishTopClass, float TopConfidence, List<PredictionScore> AllScores)> RunClassificationAsync(Image<Rgb24> originalImage, Rectangle cropBox)
        {
            if (_classificationSession == null) throw new InvalidOperationException("분류 세션 없음");
            return await Task.Run(() =>
            {
                using (var image = originalImage.Clone(x => x.Crop(cropBox).Resize(new ResizeOptions { Size = new SixLabors.ImageSharp.Size(ClassificationInputSize, ClassificationInputSize), Mode = SixLabors.ImageSharp.Processing.ResizeMode.Crop })))
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
                        var probabilities = Softmax(output.ToArray());
                        var allScores = new List<PredictionScore>();

                        // 안전장치: 배열 길이 중 작은 쪽까지만 반복
                        int loopLimit = Math.Min(probabilities.Length, _classificationClassNames.Length);
                        for (int i = 0; i < loopLimit; i++)
                        {
                            string name = _classificationClassNames[i];
                            allScores.Add(new PredictionScore { ClassName = _translationMap.GetValueOrDefault(name, name), Confidence = probabilities[i] });
                        }

                        float maxConf = probabilities.Max();
                        int maxIndex = Array.IndexOf(probabilities, maxConf);
                        string eng = (maxIndex >= 0 && maxIndex < _classificationClassNames.Length) ? _classificationClassNames[maxIndex] : "Unknown";

                        return (_translationMap.GetValueOrDefault(eng, eng), eng, maxConf, allScores);
                    }
                }
            });
        }

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
