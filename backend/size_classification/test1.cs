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
    public class TestLogItem
    {
        public string FileName { get; set; } = "";
        public BitmapImage? CroppedThumbnail { get; set; }
        public string SimulatedSize { get; set; } = ""; // 정답(Ground Truth)
        public string MeasuredArea { get; set; } = "";
        public string ResultGrade { get; set; } = "";   // 예측(Prediction)
        public string AccuracyStatus { get; set; } = ""; // 정답/오답
        public WpfBrush ResultColor { get; set; } = WpfBrushes.White;
    }

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

        // =================================================================================
        // [수정됨] 대규모 크기 시뮬레이션 테스트 (새로운 기준표 적용)
        // =================================================================================
        private async void SizeTestButton_Click(object sender, RoutedEventArgs e)
        {
            if (!CheckModelsLoaded()) return;

            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png",
                Title = "테스트할 폴더 내의 이미지 하나를 선택하세요 (100개 파일 테스트)"
            };

            if (openFileDialog.ShowDialog() != true) return;

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
            SizeTestStatusTextBlock.Text = "대규모 시뮬레이션 진행 중...";

            // [시뮬레이션 목표 면적 재설정 - 중간값]
            var testTargets = new (string Grade, double TargetArea)[]
            {
                ("소", 35000),   // 45,000 미만 (안전값)
                ("중", 60000),   // 46,000 ~ 80,000 (중간값)
                ("대", 105000),  // 80,000 ~ 130,000 (중간값)
                ("특대", 150000) // 130,000 초과 (안전값)
            };

            List<TestLogItem> sizeLogs = new List<TestLogItem>();
            int totalTests = 0;
            int totalCorrect = 0;

            Dictionary<string, (int Correct, int Total)> gradeStats = new Dictionary<string, (int, int)>
            {
                { "소", (0, 0) }, { "중", (0, 0) }, { "대", (0, 0) }, { "특대", (0, 0) }
            };

            int canvasSize = 640;

            foreach (var imagePath in imageFiles)
            {
                try
                {
                    // 1. 원본 로드 및 기준 객체 탐지
                    using var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);
                    var initialDetections = await RunDetectionAsync(imagePath);

                    if (initialDetections == null || !initialDetections.Any()) continue;

                    var baseDetection = initialDetections.OrderByDescending(r => r.Confidence).First();
                    var baseBox = baseDetection.Box;
                    baseBox.Intersect(new ImageSharpRectangle(0, 0, originalImage.Width, originalImage.Height));

                    if (baseBox.Width <= 0 || baseBox.Height <= 0) continue;

                    using var baseMangoCrop = originalImage.Clone(x => x.Crop(baseBox));
                    double aspectRatio = (double)baseBox.Width / baseBox.Height;

                    // 2. 4가지 크기로 시뮬레이션
                    foreach (var target in testTargets)
                    {
                        totalTests++;

                        int newHeight = (int)Math.Sqrt(target.TargetArea / aspectRatio);
                        int newWidth = (int)(newHeight * aspectRatio);
                        if (newWidth > canvasSize) { newWidth = canvasSize; newHeight = (int)(newWidth / aspectRatio); }
                        if (newHeight > canvasSize) { newHeight = canvasSize; newWidth = (int)(newHeight * aspectRatio); }

                        using (var simulationCanvas = new SixLabors.ImageSharp.Image<Rgb24>(canvasSize, canvasSize, SixLabors.ImageSharp.Color.Black))
                        using (var resizedMango = baseMangoCrop.Clone(x => x.Resize(newWidth, newHeight)))
                        {
                            int posX = (canvasSize - newWidth) / 2;
                            int posY = (canvasSize - newHeight) / 2;
                            simulationCanvas.Mutate(x => x.DrawImage(resizedMango, new ImageSharpPoint(posX, posY), 1f));

                            string tempPath = IOPath.Combine(IOPath.GetTempPath(), $"sim_{target.Grade}_{IOPath.GetFileName(imagePath)}");
                            simulationCanvas.Save(tempPath);

                            // 3. UI 업데이트
                            BitmapImage bitmap = new BitmapImage();
                            bitmap.BeginInit();
                            bitmap.UriSource = new Uri(tempPath, UriKind.Absolute);
                            bitmap.CacheOption = BitmapCacheOption.OnLoad;
                            bitmap.EndInit();
                            bitmap.Freeze();

                            SourceImage.Source = bitmap;
                            PreviewGrid.Width = canvasSize; PreviewGrid.Height = canvasSize;
                            await Task.Delay(5);

                            // 4. 재분석
                            var (decisionText, defects, croppedBitmap) = await RunFullPipelineAsync(tempPath, bitmap);

                            // 5. 결과 검증
                            string detectedSizeText = DetectedSizeTextBlock.Text;
                            string detectedGrade = detectedSizeText.Split(' ')[0];

                            bool isCorrect = (detectedGrade == target.Grade);

                            var currentStat = gradeStats[target.Grade];
                            gradeStats[target.Grade] = (currentStat.Correct + (isCorrect ? 1 : 0), currentStat.Total + 1);
                            if (isCorrect) totalCorrect++;

                            sizeLogs.Add(new TestLogItem
                            {
                                FileName = IOPath.GetFileName(imagePath),
                                CroppedThumbnail = croppedBitmap,
                                SimulatedSize = target.Grade,
                                MeasuredArea = $"{newWidth * newHeight:N0} px²",
                                ResultGrade = detectedGrade,
                                AccuracyStatus = isCorrect ? "정답" : "오답",
                                ResultColor = isCorrect ? WpfBrushes.LightGreen : WpfBrushes.Tomato
                            });
                        }
                    }

                    double progress = (double)totalTests / (imageFiles.Count * 4) * 100.0;
                    SizeTestStatusTextBlock.Text = $"진행 중: {progress:F1}% ({totalTests}건 완료)";
                }
                catch { continue; }
            }

            ShowSizeTestResultWindow(sizeLogs, totalTests, totalCorrect, gradeStats);
            SizeTestStatusTextBlock.Text = "테스트 완료";
        }

        // [수정됨] 결과 창 - 판단 기준(분포) 시각화 추가
        private void ShowSizeTestResultWindow(List<TestLogItem> logs, int totalCount, int correctCount, Dictionary<string, (int Correct, int Total)> gradeStats)
        {
            Window resultWindow = new Window
            {
                Title = "크기 분류 성능 분석 리포트",
                Width = 1200,
                Height = 900, // 더 넓게
                WindowStartupLocation = WindowStartupLocation.CenterScreen,
                Background = new SolidColorBrush(WpfColor.FromRgb(30, 30, 30)),
                Foreground = WpfBrushes.White,
                ResizeMode = System.Windows.ResizeMode.NoResize
            };

            Grid rootGrid = new Grid { Margin = new Thickness(20) };
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 타이틀
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 정확도 요약
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 기준 분포 그래프
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); // 리스트
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 버튼

            // 1. 타이틀
            TextBlock title = new TextBlock { Text = "크기/중량 분류 시뮬레이션 결과", FontSize = 26, FontWeight = FontWeights.Bold, HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(0, 0, 0, 10) };
            rootGrid.Children.Add(title);
            Grid.SetRow(title, 0);

            // 2. 정확도 요약
            double totalAccuracy = totalCount > 0 ? (double)correctCount / totalCount * 100.0 : 0;
            TextBlock score = new TextBlock
            {
                Text = $"종합 정확도: {totalAccuracy:F1}% ({correctCount}/{totalCount})",
                FontSize = 22,
                FontWeight = FontWeights.Bold,
                Foreground = totalAccuracy >= 90 ? WpfBrushes.LightGreen : WpfBrushes.Orange,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 0, 0, 20)
            };
            rootGrid.Children.Add(score);
            Grid.SetRow(score, 1);

            // 3. 기준 분포 그래프 (핵심 증명 자료)
            GroupBox groupStats = new GroupBox
            {
                Header = " ■ 등급별 판정 기준 및 결과 분포 (판정 근거)",
                Foreground = WpfBrushes.LightGray,
                BorderBrush = WpfBrushes.Gray,
                Margin = new Thickness(0, 0, 0, 20),
                Padding = new Thickness(10)
            };

            StackPanel statsPanel = new StackPanel();

            // 등급별 기준 정보
            string[] grades = { "소", "중", "대", "특대" };
            string[] criteria = { "< 45,000 px²", "46,000 ~ 80,000 px²", "80,000 ~ 130,000 px²", "> 130,000 px²" }; // 표 기준

            for (int i = 0; i < 4; i++)
            {
                var stat = gradeStats[grades[i]];
                double acc = stat.Total > 0 ? (double)stat.Correct / stat.Total * 100.0 : 0;

                Grid row = new Grid { Margin = new Thickness(0, 5, 0, 5) };
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(250) }); // 등급 + 기준
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) }); // 바
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(80) }); // 퍼센트
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(80) }); // 개수

                // 1) 라벨
                TextBlock label = new TextBlock
                {
                    Text = $"{grades[i]} ({criteria[i]})",
                    Foreground = WpfBrushes.White,
                    FontWeight = FontWeights.SemiBold,
                    VerticalAlignment = VerticalAlignment.Center
                };

                // 2) 바 그래프
                Grid barContainer = new Grid { Height = 20, Margin = new Thickness(10, 0, 10, 0) };
                Border bgBar = new Border { Background = new SolidColorBrush(WpfColor.FromRgb(60, 60, 60)), CornerRadius = new CornerRadius(3) };

                // 비율 그래프 (Grid Column 비율 이용)
                Grid ratioGrid = new Grid();
                // acc가 0 또는 NaN일 경우 처리
                double validAcc = double.IsNaN(acc) ? 0 : acc;

                ratioGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(validAcc, GridUnitType.Star) });
                ratioGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(100 - validAcc, GridUnitType.Star) });

                Border correctPart = new Border { Background = validAcc >= 90 ? WpfBrushes.LightGreen : WpfBrushes.Orange, CornerRadius = new CornerRadius(3, 0, 0, 3) };
                if (validAcc >= 100) correctPart.CornerRadius = new CornerRadius(3);

                ratioGrid.Children.Add(correctPart);
                Grid.SetColumn(correctPart, 0);

                barContainer.Children.Add(bgBar);
                barContainer.Children.Add(ratioGrid);

                // 3) 수치
                TextBlock perText = new TextBlock { Text = $"{validAcc:F1}%", Foreground = WpfBrushes.White, VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Right };
                TextBlock cntText = new TextBlock { Text = $"{stat.Correct}/{stat.Total}", Foreground = WpfBrushes.Gray, VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Right };

                row.Children.Add(label); Grid.SetColumn(label, 0);
                row.Children.Add(barContainer); Grid.SetColumn(barContainer, 1);
                row.Children.Add(perText); Grid.SetColumn(perText, 2);
                row.Children.Add(cntText); Grid.SetColumn(cntText, 3);

                statsPanel.Children.Add(row);
            }
            groupStats.Content = statsPanel;
            rootGrid.Children.Add(groupStats);
            Grid.SetRow(groupStats, 2);

            // 4. 상세 리스트
            DockPanel listPanel = new DockPanel();
            TextBlock listHeader = new TextBlock { Text = "■ 상세 결과 내역", FontSize = 16, FontWeight = FontWeights.Bold, Foreground = WpfBrushes.LightGray, Margin = new Thickness(0, 0, 0, 10) };
            DockPanel.SetDock(listHeader, Dock.Top);
            listPanel.Children.Add(listHeader);

            ListView logListView = new ListView
            {
                Background = WpfBrushes.Transparent,
                BorderThickness = new Thickness(1),
                BorderBrush = new SolidColorBrush(WpfColor.FromRgb(60, 60, 60)),
                Foreground = WpfBrushes.White,
                ItemsSource = logs,
                VerticalContentAlignment = VerticalAlignment.Center
            };

            GridView gridView = new GridView();

            GridViewColumn fileCol = new GridViewColumn { Header = "파일명", Width = 150 };
            var fileFactory = new FrameworkElementFactory(typeof(TextBlock));
            fileFactory.SetBinding(TextBlock.TextProperty, new Binding("FileName"));
            fileFactory.SetValue(TextBlock.ForegroundProperty, WpfBrushes.LightGray);
            fileCol.CellTemplate = new DataTemplate { VisualTree = fileFactory };

            GridViewColumn imageCol = new GridViewColumn { Header = "분석 이미지", Width = 100 };
            var imageTemplate = new DataTemplate();
            var borderFactory = new FrameworkElementFactory(typeof(Border));
            borderFactory.SetValue(Border.WidthProperty, 80.0);
            borderFactory.SetValue(Border.HeightProperty, 80.0);
            var imageFactory = new FrameworkElementFactory(typeof(System.Windows.Controls.Image));
            imageFactory.SetBinding(System.Windows.Controls.Image.SourceProperty, new Binding("CroppedThumbnail"));
            imageFactory.SetValue(System.Windows.Controls.Image.StretchProperty, Stretch.Uniform);
            borderFactory.AppendChild(imageFactory);
            imageTemplate.VisualTree = borderFactory;
            imageCol.CellTemplate = imageTemplate;

            GridViewColumn simCol = new GridViewColumn { Header = "정답 (목표)", Width = 100 };
            var simFactory = new FrameworkElementFactory(typeof(TextBlock));
            simFactory.SetBinding(TextBlock.TextProperty, new Binding("SimulatedSize"));
            simFactory.SetValue(TextBlock.FontWeightProperty, FontWeights.Bold);
            simFactory.SetValue(TextBlock.VerticalAlignmentProperty, VerticalAlignment.Center);
            simCol.CellTemplate = new DataTemplate { VisualTree = simFactory };

            GridViewColumn areaCol = new GridViewColumn { Header = "측정 면적 (증거)", Width = 150 };
            var areaFactory = new FrameworkElementFactory(typeof(TextBlock));
            areaFactory.SetBinding(TextBlock.TextProperty, new Binding("MeasuredArea"));
            areaFactory.SetValue(TextBlock.VerticalAlignmentProperty, VerticalAlignment.Center);
            areaCol.CellTemplate = new DataTemplate { VisualTree = areaFactory };

            GridViewColumn resCol = new GridViewColumn { Header = "AI 예측", Width = 250 };
            var resFactory = new FrameworkElementFactory(typeof(TextBlock));
            resFactory.SetBinding(TextBlock.TextProperty, new Binding("ResultGrade"));
            resFactory.SetValue(TextBlock.VerticalAlignmentProperty, VerticalAlignment.Center);
            resCol.CellTemplate = new DataTemplate { VisualTree = resFactory };

            GridViewColumn statCol = new GridViewColumn { Header = "판정", Width = 80 };
            var statFactory = new FrameworkElementFactory(typeof(TextBlock));
            statFactory.SetBinding(TextBlock.TextProperty, new Binding("AccuracyStatus"));
            statFactory.SetBinding(TextBlock.ForegroundProperty, new Binding("ResultColor"));
            statFactory.SetValue(TextBlock.FontWeightProperty, FontWeights.Bold);
            statFactory.SetValue(TextBlock.VerticalAlignmentProperty, VerticalAlignment.Center);
            statCol.CellTemplate = new DataTemplate { VisualTree = statFactory };

            gridView.Columns.Add(fileCol);
            gridView.Columns.Add(imageCol);
            gridView.Columns.Add(simCol);
            gridView.Columns.Add(areaCol);
            gridView.Columns.Add(resCol);
            gridView.Columns.Add(statCol);
            logListView.View = gridView;

            listPanel.Children.Add(logListView);
            rootGrid.Children.Add(listPanel);
            Grid.SetRow(listPanel, 3);

            Button closeBtn = new Button
            {
                Content = "닫기",
                Width = 100,
                Height = 35,
                Margin = new Thickness(0, 20, 0, 0),
                Background = new SolidColorBrush(WpfColor.FromRgb(0, 122, 204)),
                Foreground = WpfBrushes.White,
                BorderThickness = new Thickness(0),
                Cursor = Cursors.Hand,
                HorizontalAlignment = HorizontalAlignment.Center
            };
            closeBtn.Click += (s, e) => resultWindow.Close();
            rootGrid.Children.Add(closeBtn);
            Grid.SetRow(closeBtn, 4);

            resultWindow.Content = rootGrid;
            resultWindow.ShowDialog();
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
                ResetRightPanelToReady(); SizeTestStatusTextBlock.Text = "크기 테스트 대기 중";
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

        private async Task<(string DecisionText, List<DetectionResult> Defects, BitmapImage? CroppedImage)> RunFullPipelineAsync(string imagePath, BitmapImage bitmap)
        {
            DetectionCanvas.Children.Clear();
            DetectionResult topDetection;
            bool detectionSucceeded;
            List<DetectionResult> defectResults;
            BitmapImage? finalCroppedBitmap = null;

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
                    DetectionResultTextBlock.Text = $"망고 (객체 탐지됨 - {topDetection.Confidence * 100:F1}%)";
                }

                var cropBox = topDetection.Box;
                cropBox.Intersect(new ImageSharpRectangle(0, 0, originalImage.Width, originalImage.Height));

                if (cropBox.Width <= 0 || cropBox.Height <= 0) return ("오류", new List<DetectionResult>(), null);

                using (var crop = originalImage.Clone(x => x.Crop(cropBox)))
                {
                    using (var memoryStream = new MemoryStream())
                    {
                        crop.SaveAsPng(memoryStream);
                        memoryStream.Seek(0, SeekOrigin.Begin);

                        finalCroppedBitmap = new BitmapImage();
                        finalCroppedBitmap.BeginInit();
                        finalCroppedBitmap.CacheOption = BitmapCacheOption.OnLoad;
                        finalCroppedBitmap.StreamSource = memoryStream;
                        finalCroppedBitmap.EndInit();
                        finalCroppedBitmap.Freeze();
                    }
                }

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

                return (decision, defectResults, finalCroppedBitmap);
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

        // [수정됨] 표 기준에 맞춰 크기 분류
        private string EstimateWeightCategory(ImageSharpRectangle box)
        {
            long area = (long)box.Width * box.Height;

            if (area < 45000) return "소 (300g 미만)";
            else if (area < 80000) return "중 (300~450g)";
            else if (area < 130000) return "대 (450~600g)";
            else return "특대 (600g 초과)";
        }

        // --- 추론 헬퍼 메서드들 ---
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
                using var img = SixLabors.ImageSharp.Image.Load<Rgb24>(path);
                var (resized, scale, px, py) = PreprocessDetectionImage(img, DetectionInputSize);
                var tensor = ImageToTensor(resized, DetectionInputSize);
                resized.Dispose();
                using var res = _detectionSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) });
                return ParseYoloOutput(res.First().AsTensor<float>(), _detectionClassNames, 0.5f, scale, px, py, 0, 0);
            });
        }

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

        private (SixLabors.ImageSharp.Image<Rgb24>, float, int, int) PreprocessDetectionImage(SixLabors.ImageSharp.Image<Rgb24> original, int target)
        {
            float scale = Math.Min((float)target / original.Width, (float)target / original.Height);
            int nw = (int)(original.Width * scale), nh = (int)(original.Height * scale);
            var resized = original.Clone(x => x.Resize(nw, nh));
            int px = (target - nw) / 2, py = (target - nh) / 2;
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
