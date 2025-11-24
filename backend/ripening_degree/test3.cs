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
using System.Diagnostics;

// 모호성 해결
using Rectangle = SixLabors.ImageSharp.Rectangle;
using SixPoint = SixLabors.ImageSharp.Point;
using WpfMessageBox = System.Windows.MessageBox;
using OpenFileDialog = Microsoft.Win32.OpenFileDialog;

namespace MangoClassifierWPF
{
    // --- 데이터 클래스 ---
    public class AnalysisHistoryItem
    {
        public BitmapImage? Thumbnail { get; set; }
        public BitmapImage? FullImageSource { get; set; }
        public double OriginalImageWidth { get; set; }
        public double OriginalImageHeight { get; set; }
        public List<DetectionResult>? MangoDetections { get; set; }
        public List<DetectionResult>? DefectDetections { get; set; }
        public string FileName { get; set; } = "";
        public string DetectionResultText { get; set; } = "";
        public string DetectedSizeText { get; set; } = "";
        public string RipenessResultText { get; set; } = "";
        public string VarietyResultText { get; set; } = "";
        public string ConfidenceText { get; set; } = "";
        public string FinalDecisionText { get; set; } = "";
        public Brush? FinalDecisionBackground { get; set; }
        public Brush? FinalDecisionBrush { get; set; }
        public IEnumerable<PredictionScore>? AllRipenessScores { get; set; }
        public string DefectListText { get; set; } = "";
        public Brush? DefectListForeground { get; set; }
        public long PerfDetectionTimeMs { get; set; }
        public long PerfClassificationTimeMs { get; set; }
        public long PerfVarietyTimeMs { get; set; }
        public long PerfDefectTimeMs { get; set; }
        public long PerfTotalTimeMs { get; set; }
    }

    public class PredictionScore { public string ClassName { get; set; } = ""; public double Confidence { get; set; } }
    public class DetectionResult { public string ClassName { get; set; } = ""; public double Confidence { get; set; } public Rectangle Box { get; set; } }

    // [신규] 이미지 특성 저장용
    public class ImageFeatures
    {
        public double MeanR { get; set; }
        public double MeanG { get; set; }
        public double MeanB { get; set; }
        public double EdgeDensity { get; set; } // %
        public double BlobAreaRatio { get; set; } // %
    }

    public partial class MainWindow : Window
    {
        private InferenceSession? _classificationSession;
        private InferenceSession? _detectionSession;
        private InferenceSession? _defectSession;
        private InferenceSession? _varietySession;

        private List<AnalysisHistoryItem> _analysisHistory = new List<AnalysisHistoryItem>();
        private bool _isHistoryLoading = false;

        // 모델 설정
        private const int ClassificationInputSize = 224;
        private readonly string[] _classificationClassNames = { "breaking-stage", "half-ripe-stage", "un-healthy", "ripe", "ripe_with_consumable_disease", "unripe" };
        private readonly Dictionary<string, string> _translationMap = new Dictionary<string, string> {
            { "half-ripe-stage", "반숙" }, { "unripe", "미숙" }, { "breaking-stage", "중숙" },
            { "ripe", "익음" }, { "un-healthy", "과숙" }, { "ripe_with_consumable_disease", "흠과" }
        };

        private const int DetectionInputSize = 640;
        private readonly string[] _detectionClassNames = { "Mango" };
        private readonly Dictionary<string, string> _detectionTranslationMap = new Dictionary<string, string> { { "Mango", "망고" }, { "Not Mango", "망고 아님" } };

        private const int DefectInputSize = 640;
        private readonly string[] _defectClassNames = { "black-spot", "brown-spot", "scab" };
        private readonly Dictionary<string, string> _defectTranslationMap = new Dictionary<string, string> {
            { "brown-spot", "갈색 반점" }, { "black-spot", "검은 반점" }, { "scab", "더뎅이병" }
        };

        private const int VarietyInputSize = 224;
        private readonly string[] _varietyClassNames = { "Alphonso", "Amrapali", "Dasheri", "Langra", "Mallika", "Neelam", "Pairi", "Ramkela", "Totapuri" };

        private Dictionary<string, int> _cumulativeStats;

        // 색상 상수
        private readonly Brush PASS_COLOR = new SolidColorBrush(System.Windows.Media.Color.FromRgb(0x2E, 0xCC, 0x71));
        private readonly Brush REJECT_COLOR = Brushes.DarkRed;
        private readonly Brush CONDITIONAL_COLOR = Brushes.DarkOrange;
        private readonly Brush HOLD_COLOR = Brushes.DarkSlateGray;
        private readonly Brush TEXT_COLOR = Brushes.White;

        public MainWindow()
        {
            InitializeComponent();
            InitializeCumulativeStats();
            UpdateStatsDisplay();
            LoadModelsAsync();
            this.WindowState = WindowState.Maximized;

            FarmEnvTextBlock.Text = "온도: 28°C\n습도: 75%";
            WeatherTextBlock.Text = "맑음, 32°C\n바람: 3m/s";
            SeasonInfoTextBlock.Text = "수확기 (7월)";
        }

        private void InitializeCumulativeStats()
        {
            _cumulativeStats = new Dictionary<string, int>();
            foreach (var koreanName in _translationMap.Values)
            {
                if (!_cumulativeStats.ContainsKey(koreanName)) _cumulativeStats.Add(koreanName, 0);
            }
        }

        private void UpdateStatsDisplay()
        {
            StringBuilder statsBuilder = new StringBuilder();
            foreach (var entry in _cumulativeStats)
            {
                statsBuilder.AppendLine($"{entry.Key}: {entry.Value} 개");
            }
            CumulativeStatsTextBlock.Text = statsBuilder.ToString();
        }

        private async void LoadModelsAsync()
        {
            try
            {
                await Task.Run(() =>
                {
                    var opts = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
                    string baseDir = AppContext.BaseDirectory;
                    _classificationSession = new InferenceSession(System.IO.Path.Combine(baseDir, "best.onnx"), opts);
                    _detectionSession = new InferenceSession(System.IO.Path.Combine(baseDir, "detection.onnx"), opts);
                    _defectSession = new InferenceSession(System.IO.Path.Combine(baseDir, "defect_detection.onnx"), opts);

                    string varietyPath = System.IO.Path.Combine(baseDir, "mango_classify.onnx");
                    if (File.Exists(varietyPath)) _varietySession = new InferenceSession(varietyPath, opts);
                });
                ResetRightPanelToReady();
            }
            catch (Exception ex) { WpfMessageBox.Show($"모델 로드 오류: {ex.Message}"); }
        }

        // --------------------------------------------------------------------
        // [★핵심] 성능 검증 버튼 (시각적 배치 테스트)
        // --------------------------------------------------------------------
        private async void PerformanceTestButton_Click(object sender, RoutedEventArgs e)
        {
            // 1. 폴더 선택 (파일 하나 선택하면 그 폴더를 사용)
            var dlg = new OpenFileDialog { Title = "테스트할 폴더 내의 아무 이미지나 하나 선택하세요", Filter = "Images|*.jpg;*.png;*.jpeg" };
            if (dlg.ShowDialog() != true) return;

            string folderPath = System.IO.Path.GetDirectoryName(dlg.FileName)!;
            string[] files = Directory.GetFiles(folderPath, "*.*")
                                      .Where(s => s.EndsWith(".jpg") || s.EndsWith(".png") || s.EndsWith(".jpeg"))
                                      .Take(100) // 최대 100개
                                      .ToArray();

            if (files.Length == 0) { WpfMessageBox.Show("이미지가 없습니다."); return; }

            // 2. 정답 클래스 확인 (일단 'unripe'로 가정, 필요시 수정 가능)
            string targetClass = "unripe"; // ★ 테스트할 폴더의 정답 클래스 (예: unripe, ripe 등)

            if (WpfMessageBox.Show($"'{folderPath}' 폴더의 이미지 {files.Length}장을 \n'{targetClass}'(으)로 가정하고 테스트를 시작할까요?",
                "성능 검증 시작", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes) return;

            // 3. 통계 변수 초기화
            int correctCount = 0;
            List<double> rMeans = new(), gMeans = new(), bMeans = new();
            List<double> edgeMeans = new(), blobMeans = new();
            string originalTitle = this.Title;

            // 4. 연속 실행 (화면에 보여주면서)
            for (int i = 0; i < files.Length; i++)
            {
                string file = files[i];
                try
                {
                    this.Title = $"[{i + 1}/{files.Length}] 테스트 진행 중...";

                    // (1) 화면에 표시 및 분석 실행
                    // ProcessImageAsync 내부에서 RunFullPipelineAsync가 호출되어 화면을 갱신합니다.
                    await ProcessImageAsync(file);

                    // (2) 잠시 대기 (사용자가 화면 변화를 볼 수 있도록)
                    await Task.Delay(200); // 0.2초 대기 (속도 조절 가능)

                    // (3) 현재 분석 결과 가져오기 (화면 갱신 후 저장된 이력의 첫 번째 아이템)
                    if (_analysisHistory.Count > 0)
                    {
                        var result = _analysisHistory[0];

                        // 정답 체크 (결과 텍스트에 정답 클래스 이름이 포함되어 있는지 확인)
                        // 예: "미숙" 텍스트가 포함되어 있으면 정답 처리
                        string targetKorean = _translationMap.FirstOrDefault(x => x.Value.Contains("미숙")).Value ?? "미숙"; // 예시
                        if (result.RipenessResultText.Contains("미숙") || result.RipenessResultText.Contains("unripe")) // 단순 비교
                        {
                            correctCount++;
                        }
                    }

                    // (4) RGB/Edge/Blob 특성 계산 (별도로 수행)
                    using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(file))
                    {
                        var features = CalculateFeatures(image);
                        rMeans.Add(features.MeanR); gMeans.Add(features.MeanG); bMeans.Add(features.MeanB);
                        edgeMeans.Add(features.EdgeDensity); blobMeans.Add(features.BlobAreaRatio);
                    }
                }
                catch { /* 개별 오류 무시 */ }
            }

            this.Title = originalTitle;

            // 5. 최종 리포트 출력
            double accuracy = (double)correctCount / files.Length * 100.0;

            string report = $"[테스트 완료]\n" +
                            $"-----------------------------\n" +
                            $"📂 폴더: {System.IO.Path.GetFileName(folderPath)}\n" +
                            $"🎯 정확도: {accuracy:F2}% ({correctCount}/{files.Length})\n" +
                            $"-----------------------------\n" +
                            $"📊 평균 특성값:\n" +
                            $" - R: {rMeans.Average():F1}\n" +
                            $" - G: {gMeans.Average():F1}\n" +
                            $" - B: {bMeans.Average():F1}\n" +
                            $" - Edge: {edgeMeans.Average():F2}%\n" +
                            $" - Blob: {blobMeans.Average():F2}%";

            WpfMessageBox.Show(report, "성능 검증 결과", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        // [신규] 이미지 특성 계산 함수
        private ImageFeatures CalculateFeatures(Image<Rgb24> image)
        {
            long sumR = 0, sumG = 0, sumB = 0, edgePixels = 0, blobPixels = 0;
            int totalPixels = image.Width * image.Height;

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < image.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        var p = row[x];
                        sumR += p.R; sumG += p.G; sumB += p.B;
                        if (p.R < 60 && p.G < 60 && p.B < 60) blobPixels++;
                    }
                }
            });

            using (var edgeImage = image.Clone(x => x.DetectEdges()))
            {
                edgeImage.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < edgeImage.Height; y++)
                    {
                        var row = accessor.GetRowSpan(y);
                        for (int x = 0; x < edgeImage.Width; x++)
                        {
                            if (row[x].R > 50) edgePixels++;
                        }
                    }
                });
            }

            return new ImageFeatures
            {
                MeanR = (double)sumR / totalPixels,
                MeanG = (double)sumG / totalPixels,
                MeanB = (double)sumB / totalPixels,
                EdgeDensity = (double)edgePixels / totalPixels * 100.0,
                BlobAreaRatio = (double)blobPixels / totalPixels * 100.0
            };
        }

        // --------------------------------------------------------------------
        // 기존 단일 시연 로직 (유지)
        // --------------------------------------------------------------------
        private async void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            if (_classificationSession == null) return;
            OpenFileDialog dlg = new OpenFileDialog { Filter = "Images|*.jpg;*.jpeg;*.png" };
            if (dlg.ShowDialog() == true) await ProcessImageAsync(dlg.FileName);
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e) => ResetToWelcomeState();

        private void WelcomePanel_DragEnter(object sender, DragEventArgs e) { e.Effects = DragDropEffects.Copy; e.Handled = true; }
        private void WelcomePanel_DragOver(object sender, DragEventArgs e) { e.Effects = DragDropEffects.Copy; e.Handled = true; }
        private async void WelcomePanel_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetData(DataFormats.FileDrop) is string[] files && files.Length > 0)
                await ProcessImageAsync(files[0]);
        }

        private void HistoryListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isHistoryLoading || e.AddedItems.Count == 0 || e.AddedItems[0] is not AnalysisHistoryItem selectedItem) return;
            _isHistoryLoading = true;
            try
            {
                WelcomePanel.Visibility = Visibility.Collapsed;
                ImagePreviewPanel.Visibility = Visibility.Visible;
                SourceImage.Source = selectedItem.FullImageSource;

                DetectionCanvas.Children.Clear();
                if (selectedItem.MangoDetections != null)
                    foreach (var box in selectedItem.MangoDetections) DrawBox(box.Box, Brushes.OrangeRed, 3, selectedItem.OriginalImageWidth, selectedItem.OriginalImageHeight);
                if (selectedItem.DefectDetections != null)
                    foreach (var box in selectedItem.DefectDetections) DrawBox(box.Box, Brushes.Yellow, 2, selectedItem.OriginalImageWidth, selectedItem.OriginalImageHeight);

                DetectionResultTextBlock.Text = selectedItem.DetectionResultText;
                DetectedSizeTextBlock.Text = selectedItem.DetectedSizeText;
                RipenessResultTextBlock.Text = selectedItem.RipenessResultText;
                VarietyResultTextBlock.Text = selectedItem.VarietyResultText;
                ConfidenceTextBlock.Text = selectedItem.ConfidenceText;

                DetectionResultTextBlock.Foreground = Brushes.Orange;
                RipenessResultTextBlock.Foreground = Brushes.DodgerBlue;

                FinalDecisionTextBlock.Text = selectedItem.FinalDecisionText;
                if (FinalDecisionTextBlock.Parent is Border db) db.Background = selectedItem.FinalDecisionBackground;

                FullResultsListView.ItemsSource = selectedItem.AllRipenessScores;
                DefectResultsTextBlock.Text = selectedItem.DefectListText;
                DefectResultsTextBlock.Foreground = selectedItem.DefectListForeground;

                PerfDetectionTime.Text = $"{selectedItem.PerfDetectionTimeMs} ms";
                PerfClassificationTime.Text = $"{selectedItem.PerfClassificationTimeMs} ms";
                PerfVarietyTime.Text = $"{selectedItem.PerfVarietyTimeMs} ms";
                PerfDefectTime.Text = $"{selectedItem.PerfDefectTimeMs} ms";
                PerfTotalTime.Text = $"{selectedItem.PerfTotalTimeMs} ms";
            }
            catch (Exception ex) { WpfMessageBox.Show($"오류: {ex.Message}"); }
            finally { _isHistoryLoading = false; }
        }

        private async Task ProcessImageAsync(string imagePath)
        {
            try
            {
                ClearAllUI();
                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit(); bitmap.UriSource = new Uri(imagePath, UriKind.Absolute); bitmap.CacheOption = BitmapCacheOption.OnLoad; bitmap.EndInit(); bitmap.Freeze();
                SourceImage.Source = bitmap;

                WelcomePanel.Visibility = Visibility.Collapsed;
                ImagePreviewPanel.Visibility = Visibility.Visible;
                DetectionResultTextBlock.Text = "탐지 중...";

                var sw = Stopwatch.StartNew();
                await RunFullPipelineAsync(imagePath, bitmap, sw);
            }
            catch (Exception ex) { WpfMessageBox.Show($"오류: {ex.Message}"); ResetToWelcomeState(); }
        }

        private void ResetToWelcomeState()
        {
            WelcomePanel.Visibility = Visibility.Visible;
            ImagePreviewPanel.Visibility = Visibility.Collapsed;
            SourceImage.Source = null;
            ResetRightPanelToReady();
        }

        private void ResetRightPanelToReady()
        {
            DetectionCanvas.Children.Clear();
            DetectionResultTextBlock.Text = "준비 완료"; DetectionResultTextBlock.Foreground = Brushes.LightGreen;
            RipenessResultTextBlock.Text = "---"; DetectedSizeTextBlock.Text = "---";
            VarietyResultTextBlock.Text = "---";
            ConfidenceTextBlock.Text = "---";
            DefectResultsTextBlock.Text = "---"; FinalDecisionTextBlock.Text = "---";
            PerfTotalTime.Text = "---"; PerfDetectionTime.Text = "---"; PerfClassificationTime.Text = "---"; PerfVarietyTime.Text = "---"; PerfDefectTime.Text = "---";
        }

        private void ClearAllUI()
        {
            DetectionCanvas.Children.Clear();
            FullResultsListView.ItemsSource = null;
        }

        private async Task RunFullPipelineAsync(string imagePath, BitmapImage bitmap, Stopwatch totalStopwatch)
        {
            // 1. 망고 탐지 (640)
            var (detResults, detTime) = await RunDetectionAsync(imagePath);
            PerfDetectionTime.Text = $"{detTime} ms";

            DetectionResult? topMango = null;
            if (detResults != null && detResults.Any())
            {
                var mangos = detResults.Where(d => d.ClassName == "Mango").OrderByDescending(d => d.Confidence).ToList();
                if (mangos.Any()) topMango = mangos.First();
            }

            if (topMango == null)
            {
                DetectionResultTextBlock.Text = "망고 없음";
                totalStopwatch.Stop();
                PerfTotalTime.Text = $"{totalStopwatch.ElapsedMilliseconds} ms";
                return;
            }

            string koDetName = _detectionTranslationMap.GetValueOrDefault(topMango.ClassName, topMango.ClassName);
            DetectionResultTextBlock.Text = $"{koDetName} ({topMango.Confidence * 100:F1}%)";
            DetectionResultTextBlock.Foreground = Brushes.Orange;

            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                var cropBox = topMango.Box;
                cropBox.Intersect(new Rectangle(0, 0, originalImage.Width, originalImage.Height));

                // 병렬 실행
                var clsTask = RunClassificationAsync(originalImage, cropBox);
                var defTask = RunDefectDetectionAsync(originalImage, cropBox);
                var varTask = RunVarietyClassificationAsync(originalImage, cropBox);

                await Task.WhenAll(clsTask, defTask, varTask);

                var (koClass, enClass, conf, scores, clsTime) = clsTask.Result;
                var (defects, defTime) = defTask.Result;
                var (varietyName, varietyConf, varietyTime) = varTask.Result;

                PerfClassificationTime.Text = $"{clsTime} ms";
                PerfDefectTime.Text = $"{defTime} ms";
                PerfVarietyTime.Text = $"{varietyTime} ms";

                // 결과 업데이트
                RipenessResultTextBlock.Text = koClass;
                VarietyResultTextBlock.Text = $"{varietyName} ({varietyConf * 100:F0}%)";
                ConfidenceTextBlock.Text = $"{conf * 100:F2} %";
                FullResultsListView.ItemsSource = scores.OrderByDescending(s => s.Confidence);

                var decision = GetFinalDecision(enClass, defects, topMango.Box);
                FinalDecisionTextBlock.Text = decision.Decision;
                FinalDecisionTextBlock.Foreground = decision.TextColor;
                if (FinalDecisionTextBlock.Parent is Border db) db.Background = decision.BackgroundColor;

                if (defects.Any())
                {
                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine($"{defects.Count}개 결함:");
                    foreach (var d in defects)
                    {
                        string dName = _defectTranslationMap.GetValueOrDefault(d.ClassName, d.ClassName);
                        sb.AppendLine($"- {dName} ({d.Confidence:P0})");
                    }
                    DefectResultsTextBlock.Text = sb.ToString();
                    DefectResultsTextBlock.Foreground = Brushes.Tomato;
                }
                else
                {
                    DefectResultsTextBlock.Text = "결함 없음";
                    DefectResultsTextBlock.Foreground = Brushes.LightGreen;
                }

                string estWeight = EstimateWeightCategory(topMango.Box);
                DetectedSizeTextBlock.Text = estWeight;

                if (_cumulativeStats.ContainsKey(koClass))
                {
                    _cumulativeStats[koClass]++;
                    UpdateStatsDisplay();
                }

                DrawBox(topMango.Box, Brushes.OrangeRed, 3, originalImage.Width, originalImage.Height);
                foreach (var d in defects) DrawBox(d.Box, Brushes.Yellow, 2, originalImage.Width, originalImage.Height);

                totalStopwatch.Stop();
                PerfTotalTime.Text = $"{totalStopwatch.ElapsedMilliseconds} ms";

                var history = new AnalysisHistoryItem
                {
                    Thumbnail = bitmap,
                    FullImageSource = bitmap,
                    OriginalImageWidth = bitmap.PixelWidth,
                    OriginalImageHeight = bitmap.PixelHeight,
                    FileName = System.IO.Path.GetFileName(imagePath),
                    MangoDetections = new List<DetectionResult> { topMango },
                    DefectDetections = defects,
                    DetectionResultText = DetectionResultTextBlock.Text,
                    DetectedSizeText = estWeight,
                    RipenessResultText = koClass,
                    VarietyResultText = VarietyResultTextBlock.Text,
                    ConfidenceText = ConfidenceTextBlock.Text,
                    FinalDecisionText = decision.Decision,
                    FinalDecisionBackground = decision.BackgroundColor,
                    FinalDecisionBrush = decision.BackgroundColor == REJECT_COLOR ? Brushes.Tomato : Brushes.LightGreen,
                    AllRipenessScores = scores,
                    DefectListText = DefectResultsTextBlock.Text,
                    DefectListForeground = DefectResultsTextBlock.Foreground,
                    PerfTotalTimeMs = totalStopwatch.ElapsedMilliseconds,
                    PerfDetectionTimeMs = detTime,
                    PerfClassificationTimeMs = clsTime,
                    PerfVarietyTimeMs = varietyTime,
                    PerfDefectTimeMs = defTime
                };
                _analysisHistory.Insert(0, history);
                HistoryListView.ItemsSource = null;
                HistoryListView.ItemsSource = _analysisHistory;
            }
        }

        // --- 모델 실행 함수들 ---

        private async Task<(List<DetectionResult>, long)> RunDetectionAsync(string imagePath)
        {
            if (_detectionSession == null) throw new Exception("Detection Null");
            return await Task.Run(() => {
                var sw = Stopwatch.StartNew();
                using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);
                var (input, scale, padX, padY) = Preprocess(image, DetectionInputSize);
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
                using var results = _detectionSession.Run(inputs);
                var output = results.First().AsTensor<float>();
                var list = ParseYoloOutput(output, _detectionClassNames, 0.5f, scale, padX, padY, 0, 0);
                sw.Stop();
                return (list, sw.ElapsedMilliseconds);
            });
        }

        private async Task<(List<DetectionResult>, long)> RunDefectDetectionAsync(Image<Rgb24> original, Rectangle cropBox)
        {
            if (_defectSession == null) throw new Exception("Defect Null");
            return await Task.Run(() => {
                var sw = Stopwatch.StartNew();
                using var crop = original.Clone(x => x.Crop(cropBox));
                var (input, scale, padX, padY) = Preprocess(crop, DefectInputSize);
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
                using var results = _defectSession.Run(inputs);
                var output = results.First().AsTensor<float>();
                var list = ParseYoloOutput(output, _defectClassNames, 0.3f, scale, padX, padY, cropBox.X, cropBox.Y);
                sw.Stop();
                return (list, sw.ElapsedMilliseconds);
            });
        }

        private async Task<(string Ko, string En, float Conf, List<PredictionScore> Scores, long Time)> RunClassificationAsync(Image<Rgb24> original, Rectangle cropBox)
        {
            if (_classificationSession == null) throw new Exception("Class Null");
            return await Task.Run(() => {
                var sw = Stopwatch.StartNew();
                using var crop = original.Clone(x => x.Crop(cropBox).Resize(ClassificationInputSize, ClassificationInputSize));
                var tensor = new DenseTensor<float>(new[] { 1, 3, ClassificationInputSize, ClassificationInputSize });
                crop.ProcessPixelRows(a => {
                    for (int y = 0; y < ClassificationInputSize; y++)
                    {
                        var row = a.GetRowSpan(y);
                        for (int x = 0; x < ClassificationInputSize; x++)
                        {
                            tensor[0, 0, y, x] = row[x].R / 255f; tensor[0, 1, y, x] = row[x].G / 255f; tensor[0, 2, y, x] = row[x].B / 255f;
                        }
                    }
                });
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };
                using var results = _classificationSession.Run(inputs);
                var probs = results.First().AsTensor<float>().ToArray();

                var scores = new List<PredictionScore>();
                for (int i = 0; i < probs.Length; i++)
                {
                    string en = _classificationClassNames[i];
                    scores.Add(new PredictionScore { ClassName = _translationMap.GetValueOrDefault(en, en), Confidence = probs[i] });
                }
                int maxIdx = Array.IndexOf(probs, probs.Max());
                string enTop = _classificationClassNames[maxIdx];
                sw.Stop();
                return (_translationMap.GetValueOrDefault(enTop, enTop), enTop, probs[maxIdx], scores, sw.ElapsedMilliseconds);
            });
        }

        private async Task<(string Variety, float Conf, long Time)> RunVarietyClassificationAsync(Image<Rgb24> original, Rectangle cropBox)
        {
            if (_varietySession == null) return ("모델 없음", 0f, 0);
            return await Task.Run(() => {
                var sw = Stopwatch.StartNew();
                using var crop = original.Clone(x => x.Crop(cropBox).Resize(VarietyInputSize, VarietyInputSize));
                var tensor = new DenseTensor<float>(new[] { 1, 3, VarietyInputSize, VarietyInputSize });
                crop.ProcessPixelRows(a => {
                    for (int y = 0; y < VarietyInputSize; y++)
                    {
                        var row = a.GetRowSpan(y);
                        for (int x = 0; x < VarietyInputSize; x++)
                        {
                            tensor[0, 0, y, x] = row[x].R / 255f; tensor[0, 1, y, x] = row[x].G / 255f; tensor[0, 2, y, x] = row[x].B / 255f;
                        }
                    }
                });
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };
                using var results = _varietySession.Run(inputs);
                var probs = results.First().AsTensor<float>().ToArray();
                int maxIdx = Array.IndexOf(probs, probs.Max());
                sw.Stop();
                return (_varietyClassNames[maxIdx], probs[maxIdx], sw.ElapsedMilliseconds);
            });
        }

        private (DenseTensor<float>, float, int, int) Preprocess(Image<Rgb24> img, int size)
        {
            float scale = Math.Min((float)size / img.Width, (float)size / img.Height);
            int w = (int)(img.Width * scale), h = (int)(img.Height * scale);
            int px = (size - w) / 2, py = (size - h) / 2;
            using var resized = img.Clone(x => x.Resize(w, h));
            using var final = new Image<Rgb24>(size, size, new Rgb24(114, 114, 114));
            final.Mutate(x => x.DrawImage(resized, new SixPoint(px, py), 1f));
            var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });
            final.ProcessPixelRows(a => {
                for (int y = 0; y < size; y++)
                {
                    var row = a.GetRowSpan(y);
                    for (int x = 0; x < size; x++)
                    {
                        tensor[0, 0, y, x] = row[x].R / 255f; tensor[0, 1, y, x] = row[x].G / 255f; tensor[0, 2, y, x] = row[x].B / 255f;
                    }
                }
            });
            return (tensor, scale, px, py);
        }

        private List<DetectionResult> ParseYoloOutput(Tensor<float> output, string[] classes, float thresh, float scale, int px, int py, int offX, int offY)
        {
            var list = new List<DetectionResult>();
            int numBoxes = output.Dimensions[2];
            int numClasses = classes.Length;
            for (int i = 0; i < numBoxes; i++)
            {
                float maxConf = 0f; int maxClass = -1;
                for (int j = 0; j < numClasses; j++)
                {
                    float c = output[0, 4 + j, i];
                    if (c > maxConf) { maxConf = c; maxClass = j; }
                }
                if (maxConf > thresh)
                {
                    float cx = output[0, 0, i], cy = output[0, 1, i], w = output[0, 2, i], h = output[0, 3, i];
                    float x = (cx - w / 2 - px) / scale + offX;
                    float y = (cy - h / 2 - py) / scale + offY;
                    list.Add(new DetectionResult { ClassName = classes[maxClass], Confidence = maxConf, Box = new Rectangle((int)x, (int)y, (int)(w / scale), (int)(h / scale)) });
                }
            }
            var result = new List<DetectionResult>();
            foreach (var item in list.OrderByDescending(l => l.Confidence))
            {
                if (!result.Any(e => Rectangle.Intersect(item.Box, e.Box).Width * Rectangle.Intersect(item.Box, e.Box).Height > 0.45 * item.Box.Width * item.Box.Height))
                    result.Add(item);
            }
            return result;
        }

        private (string Decision, Brush TextColor, Brush BackgroundColor) GetFinalDecision(string ripeness, List<DetectionResult> defects, Rectangle box)
        {
            if (defects.Any(d => d.ClassName == "scab")) return ("판매 금지 (병해)", TEXT_COLOR, REJECT_COLOR);
            if (ripeness == "un-healthy") return ("판매 금지 (과숙)", TEXT_COLOR, REJECT_COLOR);
            if (defects.Any(d => d.ClassName == "black-spot")) return ("제한적 (가공용)", TEXT_COLOR, CONDITIONAL_COLOR);

            bool hasBrown = defects.Any(d => d.ClassName == "brown-spot");
            if (ripeness == "ripe" || ripeness == "half-ripe-stage") return ("판매 가능", TEXT_COLOR, PASS_COLOR);
            return ("보류", TEXT_COLOR, HOLD_COLOR);
        }

        private string EstimateWeightCategory(Rectangle box)
        {
            long area = box.Width * box.Height;
            if (area < 50000) return "소";
            if (area < 100000) return "중";
            if (area < 150000) return "대";
            return "특대";
        }

        private void DrawBox(Rectangle box, Brush brush, double thickness, double orgW, double orgH)
        {
            if (orgW == 0 || orgH == 0) return;
            double scale = Math.Min(PreviewGrid.ActualWidth / orgW, PreviewGrid.ActualHeight / orgH);
            double offX = (PreviewGrid.ActualWidth - orgW * scale) / 2;
            double offY = (PreviewGrid.ActualHeight - orgH * scale) / 2;
            var rect = new System.Windows.Shapes.Rectangle { Stroke = brush, StrokeThickness = thickness, Width = box.Width * scale, Height = box.Height * scale };
            Canvas.SetLeft(rect, box.X * scale + offX); Canvas.SetTop(rect, box.Y * scale + offY);
            DetectionCanvas.Children.Add(rect);
        }
    }
}
