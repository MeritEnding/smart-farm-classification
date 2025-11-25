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
    public class AnalysisHistoryItem
    {
        public BitmapImage? Thumbnail { get; set; }
        public BitmapImage? FullImageSource { get; set; }
        public double OriginalImageWidth { get; set; }
        public double OriginalImageHeight { get; set; }

        public Rectangle AnalysisBox { get; set; }
        public List<DetectionResult>? DefectDetections { get; set; }

        public string FileName { get; set; } = "";

        // UI 표시용
        public string DetectionResultText { get; set; } = "";
        public string DetectedSizeText { get; set; } = "";
        public string RipenessResultText { get; set; } = "";
        public string VarietyResultText { get; set; } = "";
        public string ConfidenceText { get; set; } = "";
        public string FinalDecisionText { get; set; } = "";

        // 통계 데이터
        public byte ValR { get; set; }
        public byte ValG { get; set; }
        public byte ValB { get; set; }
        public double ValEdge { get; set; }
        public int ValBlobCount { get; set; }

        // [추가] 어떤 결함들이었는지 저장 (통계용)
        public List<string> ValDefectTypes { get; set; } = new List<string>();

        public Brush? FinalDecisionBackground { get; set; }
        public Brush? FinalDecisionBrush { get; set; }
        public IEnumerable<PredictionScore>? AllRipenessScores { get; set; }
        public string DefectListText { get; set; } = "";
        public Brush? DefectListForeground { get; set; }

        // 성능 시간
        public long PerfClassificationTimeMs { get; set; }
        public long PerfVarietyTimeMs { get; set; }
        public long PerfDefectTimeMs { get; set; }
        public long PerfTotalTimeMs { get; set; }
    }

    public class PredictionScore { public string ClassName { get; set; } = ""; public double Confidence { get; set; } }
    public class DetectionResult { public string ClassName { get; set; } = ""; public double Confidence { get; set; } public Rectangle Box { get; set; } }

    public partial class MainWindow : Window
    {
        private InferenceSession? _classificationSession;
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

        private const int DefectInputSize = 640;
        private readonly string[] _defectClassNames = { "black-spot", "brown-spot", "scab" };
        private readonly Dictionary<string, string> _defectTranslationMap = new Dictionary<string, string> {
            { "brown-spot", "갈색 반점" }, { "black-spot", "검은 반점" }, { "scab", "더뎅이병" }
        };

        private const int VarietyInputSize = 224;
        private readonly string[] _varietyClassNames = { "Alphonso", "Amrapali", "Dasheri", "Langra", "Mallika", "Neelam", "Pairi", "Ramkela", "Totapuri" };

        private Dictionary<string, int> _cumulativeStats = new Dictionary<string, int>();

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

            PerfDetectionTime.Text = "미사용";
        }

        private void InitializeCumulativeStats()
        {
            _cumulativeStats.Clear();
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
                    _defectSession = new InferenceSession(System.IO.Path.Combine(baseDir, "defect_detection.onnx"), opts);

                    string varietyPath = System.IO.Path.Combine(baseDir, "mango_classify.onnx");
                    if (File.Exists(varietyPath)) _varietySession = new InferenceSession(varietyPath, opts);
                });
                ResetRightPanelToReady();
            }
            catch (Exception ex) { WpfMessageBox.Show($"모델 로드 오류: {ex.Message}"); }
        }

        // =========================================================
        // [수정] 성능 검증: 결함 상세 통계 및 그래프 개선
        // =========================================================
        private async void PerformanceTestButton_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Title = "테스트할 폴더 선택 (이미지 파일 하나 선택)", Filter = "Images|*.jpg;*.png;*.jpeg" };
            if (dlg.ShowDialog() != true) return;

            string folderPath = System.IO.Path.GetDirectoryName(dlg.FileName)!;
            string[] files = Directory.GetFiles(folderPath, "*.*")
                                      .Where(s => s.EndsWith(".jpg") || s.EndsWith(".png") || s.EndsWith(".jpeg"))
                                      .Take(100)
                                      .ToArray();

            if (files.Length == 0) { WpfMessageBox.Show("이미지가 없습니다."); return; }

            string targetClassEnglish = "ripe";
            if (WpfMessageBox.Show($"'{folderPath}' 폴더의 이미지 {files.Length}장을\n[ 정답: {targetClassEnglish} ]로 가정하고 테스트합니까?",
                "검증", MessageBoxButton.YesNo) != MessageBoxResult.Yes) return;

            string targetClassKorean = _translationMap.ContainsKey(targetClassEnglish) ? _translationMap[targetClassEnglish] : targetClassEnglish;

            int correctCount = 0;
            int validImages = 0;

            // 값들을 리스트에 저장
            List<byte> listR = new List<byte>();
            List<byte> listG = new List<byte>();
            List<byte> listB = new List<byte>();
            List<double> listEdge = new List<double>();
            List<int> listBlobs = new List<int>();
            List<string> allDefectTypes = new List<string>(); // 발견된 모든 결함 이름 수집

            string originalTitle = this.Title;

            for (int i = 0; i < files.Length; i++)
            {
                try
                {
                    this.Title = $"[{i + 1}/{files.Length}] 테스트 진행 중...";
                    await ProcessImageAsync(files[i]);
                    await Task.Delay(20);

                    if (_analysisHistory.Count > 0)
                    {
                        var lastResult = _analysisHistory[0];
                        if (lastResult.FileName == System.IO.Path.GetFileName(files[i]))
                        {
                            validImages++;

                            if (lastResult.RipenessResultText == targetClassKorean)
                                correctCount++;

                            listR.Add(lastResult.ValR);
                            listG.Add(lastResult.ValG);
                            listB.Add(lastResult.ValB);
                            listEdge.Add(lastResult.ValEdge);
                            listBlobs.Add(lastResult.ValBlobCount);
                            allDefectTypes.AddRange(lastResult.ValDefectTypes); // 결함 종류 누적
                        }
                    }
                }
                catch { }
            }

            this.Title = originalTitle;

            double accuracy = validImages > 0 ? (double)correctCount / validImages * 100.0 : 0;
            if (validImages == 0) { WpfMessageBox.Show("분석된 이미지가 없습니다."); return; }

            // [그래프 그리기 함수]
            string DrawHistogram(string name, List<byte> values, int step)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"--- [{name} 분포 (간격:{step})] ---");

                for (int start = 0; start <= 250; start += step)
                {
                    int end = (start + step - 1) > 255 ? 255 : (start + step - 1);
                    int count = values.Count(v => v >= start && v <= end);

                    if (count > 0)
                    {
                        string bar = new string('■', Math.Min(count, 20));
                        sb.AppendLine($" {start:D3}~{end:D3}: {bar} ({count})");
                    }
                }
                return sb.ToString();
            }

            string DrawEdgeHistogram(List<double> values)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"--- [Edge(거칠기) 분포] ---");
                int[] ranges = { 0, 10, 20, 30, 40, 50, 100 };
                for (int j = 0; j < ranges.Length - 1; j++)
                {
                    int s = ranges[j];
                    int e = ranges[j + 1];
                    int count = values.Count(v => v >= s && v < e);
                    if (count > 0)
                    {
                        string bar = new string('■', Math.Min(count, 20));
                        sb.AppendLine($" {s:D2} ~ {e:D2} : {bar} ({count})");
                    }
                }
                return sb.ToString();
            }

            // [리포트 생성]
            StringBuilder report = new StringBuilder();
            report.AppendLine($"[ 종합 성능 결과 리포트 ]");
            report.AppendLine($"----------------------------------------");
            report.AppendLine($"📂 폴더: {System.IO.Path.GetFileName(folderPath)}");
            report.AppendLine($"🎯 총 {validImages}장 분석 완료");
            report.AppendLine($"✅ 정확도: {accuracy:F1}% ({correctCount}/{validImages})");
            report.AppendLine($"----------------------------------------");

            report.AppendLine($"📊 [기본 통계]");
            report.AppendLine($" R 평균: {listR.Average(v => (double)v):F0} (Min:{listR.Min()} ~ Max:{listR.Max()})");
            report.AppendLine($" G 평균: {listG.Average(v => (double)v):F0} (Min:{listG.Min()} ~ Max:{listG.Max()})");
            report.AppendLine($" B 평균: {listB.Average(v => (double)v):F0} (Min:{listB.Min()} ~ Max:{listB.Max()})");
            report.AppendLine($" Edge : {listEdge.Average():F1} (Min:{listEdge.Min():F1} ~ Max:{listEdge.Max():F1})");

            // Blob 통계 상세화
            report.AppendLine($" Blob : 평균 {listBlobs.Average():F1}개 발견 (최대 {listBlobs.Max()}개)");
            report.AppendLine($"----------------------------------------");

            // 20 단위 히스토그램
            report.AppendLine(DrawHistogram("Red", listR, 20));
            report.AppendLine(DrawHistogram("Green", listG, 20));
            report.AppendLine(DrawHistogram("Blue", listB, 20));
            report.AppendLine(DrawEdgeHistogram(listEdge));

            // [수정] Blob 그래프 - 이미지 수 기준
            report.AppendLine("--- [Blob(결함) 발견 이미지 분포] ---");
            var blobGroups = listBlobs.GroupBy(x => x).OrderBy(g => g.Key);
            foreach (var g in blobGroups)
            {
                string bar = new string('■', Math.Min(g.Count(), 20));
                // "3개 발견된 이미지: 5장" 형태로 명확히 표시
                report.AppendLine($" {g.Key}개 발견된 사진: {bar} ({g.Count()}장)");
            }

            // [추가] 결함 유형별 총 개수 (누적)
            report.AppendLine("");
            report.AppendLine("--- [발견된 결함 종류 총합] ---");
            if (allDefectTypes.Count > 0)
            {
                var typeGroups = allDefectTypes.GroupBy(x => x).OrderByDescending(g => g.Count());
                foreach (var g in typeGroups)
                {
                    report.AppendLine($" • {g.Key}: 총 {g.Count()}개");
                }
            }
            else
            {
                report.AppendLine(" • 발견된 결함 없음");
            }

            WpfMessageBox.Show(report.ToString(), "상세 분석 결과", MessageBoxButton.OK, MessageBoxImage.Information);
        }

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

                DrawBox(selectedItem.AnalysisBox, Brushes.Cyan, 2, selectedItem.OriginalImageWidth, selectedItem.OriginalImageHeight);

                if (selectedItem.DefectDetections != null)
                    foreach (var box in selectedItem.DefectDetections) DrawBox(box.Box, Brushes.Yellow, 2, selectedItem.OriginalImageWidth, selectedItem.OriginalImageHeight);

                DetectionResultTextBlock.Text = selectedItem.DetectionResultText;
                RipenessResultTextBlock.Text = selectedItem.RipenessResultText;
                VarietyResultTextBlock.Text = selectedItem.VarietyResultText;
                ConfidenceTextBlock.Text = selectedItem.ConfidenceText;

                DetectedSizeTextBlock.Text = selectedItem.DetectedSizeText;

                DetectionResultTextBlock.Foreground = Brushes.LightSkyBlue;
                RipenessResultTextBlock.Foreground = Brushes.DodgerBlue;

                FinalDecisionTextBlock.Text = selectedItem.FinalDecisionText;
                if (FinalDecisionTextBlock.Parent is Border db) db.Background = selectedItem.FinalDecisionBackground;

                FullResultsListView.ItemsSource = selectedItem.AllRipenessScores;
                DefectResultsTextBlock.Text = selectedItem.DefectListText;
                DefectResultsTextBlock.Foreground = selectedItem.DefectListForeground;

                PerfDetectionTime.Text = "미사용";
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
                DetectionResultTextBlock.Text = "분석 중...";

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
            PerfTotalTime.Text = "---"; PerfDetectionTime.Text = "미사용"; PerfClassificationTime.Text = "---"; PerfVarietyTime.Text = "---"; PerfDefectTime.Text = "---";
        }

        private void ClearAllUI()
        {
            DetectionCanvas.Children.Clear();
            FullResultsListView.ItemsSource = null;
        }

        private async Task RunFullPipelineAsync(string imagePath, BitmapImage bitmap, Stopwatch totalStopwatch)
        {
            int imgW = bitmap.PixelWidth;
            int imgH = bitmap.PixelHeight;

            int boxW = (int)(imgW * 0.55);
            int boxH = (int)(imgH * 0.55);
            int startX = (imgW - boxW) / 2;
            int startY = (imgH - boxH) / 2;

            Rectangle cropBox = new Rectangle(startX, startY, boxW, boxH);

            DetectionResultTextBlock.Text = "중앙 영역 분석";
            DetectionResultTextBlock.Foreground = Brushes.LightSkyBlue;

            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                var (r, g, b, edgeScore) = AnalyzeCropFeatures(originalImage, cropBox);

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

                RipenessResultTextBlock.Text = koClass;
                VarietyResultTextBlock.Text = $"{varietyName} ({varietyConf * 100:F0}%)";
                ConfidenceTextBlock.Text = $"{conf * 100:F2} %";
                FullResultsListView.ItemsSource = scores.OrderByDescending(s => s.Confidence);

                var decision = GetFinalDecision(enClass, defects);
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

                string statsText = $"[특성 데이터]\n" +
                                   $"분석: 중앙 {boxW}x{boxH}\n" +
                                   $"RGB: {r}, {g}, {b}\n" +
                                   $"Edge: {edgeScore:F1} (거칠기)\n" +
                                   $"Blob: {defects.Count}개 (결함)";

                DetectedSizeTextBlock.Text = statsText;

                if (_cumulativeStats.ContainsKey(koClass))
                {
                    _cumulativeStats[koClass]++;
                    UpdateStatsDisplay();
                }

                DrawBox(cropBox, Brushes.Cyan, 2, originalImage.Width, originalImage.Height);
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

                    AnalysisBox = cropBox,
                    DefectDetections = defects,
                    DetectionResultText = "중앙 영역 분석",

                    DetectedSizeText = statsText,

                    ValR = r,
                    ValG = g,
                    ValB = b,
                    ValEdge = edgeScore,
                    ValBlobCount = defects.Count,

                    // [수정] 결함 종류 리스트 저장 (한글 이름으로 변환해서)
                    ValDefectTypes = defects.Select(d => _defectTranslationMap.GetValueOrDefault(d.ClassName, d.ClassName)).ToList(),

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
                    PerfClassificationTimeMs = clsTime,
                    PerfVarietyTimeMs = varietyTime,
                    PerfDefectTimeMs = defTime
                };
                _analysisHistory.Insert(0, history);
                HistoryListView.ItemsSource = null;
                HistoryListView.ItemsSource = _analysisHistory;
            }
        }

        private (byte R, byte G, byte B, double EdgeScore) AnalyzeCropFeatures(Image<Rgb24> original, Rectangle cropBox)
        {
            using var crop = original.Clone(x => x.Crop(cropBox));

            double rSum = 0, gSum = 0, bSum = 0;
            int pixelCount = crop.Width * crop.Height;

            crop.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        rSum += row[x].R;
                        gSum += row[x].G;
                        bSum += row[x].B;
                    }
                }
            });

            byte avgR = (byte)(rSum / pixelCount);
            byte avgG = (byte)(gSum / pixelCount);
            byte avgB = (byte)(bSum / pixelCount);

            using var edgeImage = crop.Clone(x => x.DetectEdges());
            double edgeSum = 0;

            edgeImage.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        edgeSum += (row[x].R + row[x].G + row[x].B) / 3.0;
                    }
                }
            });

            double edgeScore = edgeSum / pixelCount;

            return (avgR, avgG, avgB, edgeScore);
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

        private (string Decision, Brush TextColor, Brush BackgroundColor) GetFinalDecision(string ripeness, List<DetectionResult> defects)
        {
            if (defects.Any(d => d.ClassName == "scab")) return ("판매 금지 (병해)", TEXT_COLOR, REJECT_COLOR);
            if (ripeness == "un-healthy") return ("판매 금지 (과숙)", TEXT_COLOR, REJECT_COLOR);
            if (defects.Any(d => d.ClassName == "black-spot")) return ("제한적 (가공용)", TEXT_COLOR, CONDITIONAL_COLOR);

            bool hasBrown = defects.Any(d => d.ClassName == "brown-spot");
            if (ripeness == "ripe" || ripeness == "half-ripe-stage") return ("판매 가능", TEXT_COLOR, PASS_COLOR);
            return ("보류", TEXT_COLOR, HOLD_COLOR);
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
