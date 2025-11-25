using Microsoft.Win32;
using System;
using System.Collections.Generic;
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
using System.Windows.Shapes; // WPF Shapes (Rectangle, Path etc)

// [충돌 방지를 위한 별칭 설정]
using IOPath = System.IO.Path;
using WpfColor = System.Windows.Media.Color;
using WpfBrushes = System.Windows.Media.Brushes;
using WpfBrush = System.Windows.Media.Brush;

// AI/ImageSharp 관련
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

// SixLabors와 WPF간의 모호함 해결
using ImageSharpRectangle = SixLabors.ImageSharp.Rectangle;
using ImageSharpPoint = SixLabors.ImageSharp.Point;

namespace MangoClassifierWPF
{
    // 테스트 상세 로그용 클래스
    public class TestLogItem
    {
        public string FileName { get; set; } = "";
        public string ResultCategory { get; set; } = ""; // 폐기, 제한적, 가능
        public string MainReason { get; set; } = "";     // 판정 주 원인 (예: 과숙)
        public string DefectInfo { get; set; } = "";     // 세부 결함 정보 (예: 검은반점 2개)
        public WpfBrush ResultColor { get; set; } = WpfBrushes.White;

        // 리스트뷰에 보여질 전체 텍스트 (바인딩용)
        public string DisplayDetail => string.IsNullOrEmpty(DefectInfo) ? MainReason : $"{MainReason} | {DefectInfo}";
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
        { "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango", "Mango" };

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

        // [정확도 테스트 버튼]
        private async void AccuracyTestButton_Click(object sender, RoutedEventArgs e)
        {
            if (!CheckModelsLoaded()) return;

            var answer = MessageBox.Show(
                "테스트할 폴더가 '폐기(불량)' 이미지들로만 구성되어 있나요?\n\n" +
                "예 (Yes) : 폐기 폴더 (모델이 '폐기'로 예측해야 정답)\n" +
                "아니요 (No) : 정상/판매가능 폴더 (모델이 '가능/제한적'으로 예측해야 정답)",
                "테스트 기준 설정",
                MessageBoxButton.YesNoCancel,
                MessageBoxImage.Question);

            if (answer == MessageBoxResult.Cancel) return;

            bool isTestingDiscardFolder = (answer == MessageBoxResult.Yes);
            string targetLabel = isTestingDiscardFolder ? "폐기(불량)" : "정상(판매가능)";

            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png",
                Title = $"{targetLabel} 폴더 내의 이미지 하나를 선택하세요"
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

                    int totalCount = 0;
                    int correctCount = 0;

                    List<TestLogItem> testLogs = new List<TestLogItem>();

                    AccuracyResultTextBlock.Text = "테스트 시작...";

                    foreach (var imagePath in imageFiles)
                    {
                        totalCount++;

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

                        // 파이프라인 실행 (판정 결과 텍스트, 결함 리스트 리턴)
                        var (decisionText, defects) = await RunFullPipelineAsync(imagePath, bitmap);

                        // [수정] 결함 개수 텍스트 생성
                        int scabs = defects.Count(d => d.ClassName == "scab");
                        int blackSpots = defects.Count(d => d.ClassName == "black-spot");
                        int brownSpots = defects.Count(d => d.ClassName == "brown-spot");

                        List<string> defectDetails = new List<string>();
                        if (scabs > 0) defectDetails.Add($"더뎅이병 {scabs}개");
                        if (blackSpots > 0) defectDetails.Add($"검은반점 {blackSpots}개");
                        if (brownSpots > 0) defectDetails.Add($"갈색반점 {brownSpots}개");
                        if (defectDetails.Count == 0) defectDetails.Add("결함 없음");

                        string defectInfoStr = string.Join(", ", defectDetails);

                        // 판정 카테고리 및 색상
                        string category = "알 수 없음";
                        WpfBrush color = WpfBrushes.Gray;

                        if (decisionText.Contains("폐기")) { category = "폐기"; color = new SolidColorBrush(WpfColor.FromRgb(255, 80, 80)); }
                        else if (decisionText.Contains("제한적")) { category = "제한적"; color = new SolidColorBrush(WpfColor.FromRgb(255, 165, 0)); }
                        else if (decisionText.Contains("가능")) { category = "가능"; color = new SolidColorBrush(WpfColor.FromRgb(46, 204, 113)); }

                        // 로그 저장
                        testLogs.Add(new TestLogItem
                        {
                            FileName = IOPath.GetFileName(imagePath),
                            ResultCategory = category,
                            MainReason = decisionText, // 판정 이유 (예: 폐기 (과숙))
                            DefectInfo = defectInfoStr, // 상세 결함 (예: 검은반점 2개, 갈색반점 1개)
                            ResultColor = color
                        });

                        // 정답 체크
                        bool isAiPredictedDiscard = decisionText.Contains("폐기");
                        bool isCorrect = isTestingDiscardFolder ? isAiPredictedDiscard : !isAiPredictedDiscard;
                        if (isCorrect) correctCount++;

                        double currentAccuracy = (double)correctCount / totalCount * 100.0;
                        AccuracyResultTextBlock.Text = $"[{totalCount}/{imageFiles.Count}] 정답: {correctCount}개 ({currentAccuracy:F1}%)";
                    }

                    // 결과 창 표시
                    ShowDistributionResultWindow(testLogs, totalCount, correctCount, targetLabel);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"테스트 오류: {ex.Message}");
                    AccuracyResultTextBlock.Text = "오류 발생";
                }
            }
        }

        // [결과 창 생성 - 상세 로그 및 분포 그래프]
        private void ShowDistributionResultWindow(
            List<TestLogItem> logs,
            int totalCount,
            int correctCount,
            string targetLabel)
        {
            Window resultWindow = new Window
            {
                Title = "테스트 상세 리포트",
                Width = 900,
                Height = 800, // 창 크기 증가
                WindowStartupLocation = WindowStartupLocation.CenterScreen,
                Background = new SolidColorBrush(WpfColor.FromRgb(30, 30, 30)),
                Foreground = WpfBrushes.White,
                ResizeMode = System.Windows.ResizeMode.NoResize
            };

            Grid rootGrid = new Grid { Margin = new Thickness(20) };
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Header
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Graph
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); // List
            rootGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Button

            // 1. 헤더
            StackPanel headerPanel = new StackPanel();
            headerPanel.Children.Add(new TextBlock { Text = $"[{targetLabel} 폴더] 테스트 결과", FontSize = 22, FontWeight = FontWeights.Bold, HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(0, 0, 0, 5) });

            double accuracy = (double)correctCount / totalCount * 100.0;
            headerPanel.Children.Add(new TextBlock
            {
                Text = $"정확도: {accuracy:F1}% ({correctCount}/{totalCount} 정답)",
                FontSize = 18,
                FontWeight = FontWeights.SemiBold,
                HorizontalAlignment = HorizontalAlignment.Center,
                Foreground = accuracy >= 90 ? WpfBrushes.LightGreen : WpfBrushes.Orange,
                Margin = new Thickness(0, 0, 0, 20)
            });
            rootGrid.Children.Add(headerPanel);
            Grid.SetRow(headerPanel, 0);

            // 2. 그래프 (사유별 분포)
            StackPanel graphContainer = new StackPanel { Margin = new Thickness(0, 0, 0, 20) };
            graphContainer.Children.Add(new TextBlock { Text = "■ 상세 판정 사유 분포", FontSize = 16, FontWeight = FontWeights.Bold, Foreground = WpfBrushes.LightGray, Margin = new Thickness(0, 0, 0, 10) });

            // 사유별 그룹핑
            var reasonGroups = logs.GroupBy(x => x.MainReason)
                                   .Select(g => new { Reason = g.Key, Count = g.Count() })
                                   .OrderByDescending(x => x.Count);

            foreach (var item in reasonGroups)
            {
                CreateBarChartRow(graphContainer, item.Reason, item.Count, totalCount);
            }
            rootGrid.Children.Add(graphContainer);
            Grid.SetRow(graphContainer, 1);

            // 3. 상세 로그 리스트 (ListView)
            DockPanel listPanel = new DockPanel();

            TextBlock listHeader = new TextBlock
            {
                Text = "■ 개별 파일 판정 내역",
                FontSize = 16,
                FontWeight = FontWeights.Bold,
                Foreground = WpfBrushes.LightGray,
                Margin = new Thickness(0, 0, 0, 10)
            };
            DockPanel.SetDock(listHeader, Dock.Top);
            listPanel.Children.Add(listHeader);

            ListView logListView = new ListView
            {
                Background = WpfBrushes.Transparent,
                BorderThickness = new Thickness(1),
                BorderBrush = new SolidColorBrush(WpfColor.FromRgb(60, 60, 60)),
                Foreground = WpfBrushes.White,
                ItemsSource = logs
            };

            GridView gridView = new GridView();

            // 파일명 컬럼
            GridViewColumn fileCol = new GridViewColumn { Header = "파일명", Width = 200 };
            var fileTemplate = new DataTemplate();
            var fileFactory = new FrameworkElementFactory(typeof(TextBlock));
            fileFactory.SetBinding(TextBlock.TextProperty, new Binding("FileName"));
            fileFactory.SetValue(TextBlock.ForegroundProperty, WpfBrushes.LightGray);
            fileTemplate.VisualTree = fileFactory;
            fileCol.CellTemplate = fileTemplate;

            // 판정결과 컬럼
            GridViewColumn resultCol = new GridViewColumn { Header = "판정", Width = 80 };
            var resultTemplate = new DataTemplate();
            var resultFactory = new FrameworkElementFactory(typeof(TextBlock));
            resultFactory.SetBinding(TextBlock.TextProperty, new Binding("ResultCategory"));
            resultFactory.SetBinding(TextBlock.ForegroundProperty, new Binding("ResultColor"));
            resultFactory.SetValue(TextBlock.FontWeightProperty, FontWeights.Bold);
            resultTemplate.VisualTree = resultFactory;
            resultCol.CellTemplate = resultTemplate;

            // [수정됨] 상세 사유 컬럼 (바인딩 DisplayDetail 사용)
            GridViewColumn reasonCol = new GridViewColumn { Header = "상세 사유 및 결함 정보", Width = 500 };
            var reasonFactory = new FrameworkElementFactory(typeof(TextBlock));
            reasonFactory.SetBinding(TextBlock.TextProperty, new Binding("DisplayDetail"));
            reasonFactory.SetValue(TextBlock.TextWrappingProperty, TextWrapping.Wrap); // 긴 텍스트 줄바꿈
            reasonCol.CellTemplate = new DataTemplate { VisualTree = reasonFactory };

            gridView.Columns.Add(fileCol);
            gridView.Columns.Add(resultCol);
            gridView.Columns.Add(reasonCol);
            logListView.View = gridView;

            listPanel.Children.Add(logListView);
            rootGrid.Children.Add(listPanel);
            Grid.SetRow(listPanel, 2);

            // 4. 닫기 버튼
            Button closeBtn = new Button
            {
                Content = "닫기",
                Width = 120,
                Height = 40,
                Margin = new Thickness(0, 20, 0, 0),
                Background = new SolidColorBrush(WpfColor.FromRgb(0, 122, 204)),
                Foreground = WpfBrushes.White,
                BorderThickness = new Thickness(0),
                FontSize = 14,
                FontWeight = FontWeights.Bold,
                Cursor = Cursors.Hand,
                HorizontalAlignment = HorizontalAlignment.Center
            };
            closeBtn.Click += (s, e) => resultWindow.Close();
            rootGrid.Children.Add(closeBtn);
            Grid.SetRow(closeBtn, 3);

            resultWindow.Content = rootGrid;
            resultWindow.ShowDialog();
        }

        // [그래프 그리기 헬퍼]
        private void CreateBarChartRow(StackPanel panel, string label, int count, int totalBase)
        {
            double percent = totalBase > 0 ? (double)count / totalBase * 100.0 : 0;

            WpfBrush barColor = WpfBrushes.Gray;
            if (label.Contains("폐기")) barColor = new SolidColorBrush(WpfColor.FromRgb(255, 80, 80));
            else if (label.Contains("제한적")) barColor = new SolidColorBrush(WpfColor.FromRgb(255, 165, 0));
            else if (label.Contains("가능")) barColor = new SolidColorBrush(WpfColor.FromRgb(46, 204, 113));

            Grid rowGrid = new Grid { Margin = new Thickness(0, 0, 0, 5) };
            rowGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(250) }); // 라벨
            rowGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) }); // 바
            rowGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(50) }); // 카운트

            TextBlock labelBlock = new TextBlock
            {
                Text = label,
                Foreground = WpfBrushes.Silver,
                FontSize = 12,
                VerticalAlignment = VerticalAlignment.Center,
                TextTrimming = TextTrimming.CharacterEllipsis,
                ToolTip = label
            };

            Grid barGrid = new Grid { Margin = new Thickness(10, 0, 10, 0), Height = 16 };
            Border bgBar = new Border { Background = new SolidColorBrush(WpfColor.FromRgb(50, 50, 50)), CornerRadius = new CornerRadius(2) };
            Border fillBar = new Border
            {
                Background = barColor,
                CornerRadius = new CornerRadius(2),
                HorizontalAlignment = HorizontalAlignment.Left,
                Width = 400 * (percent / 100.0)
            };
            if (fillBar.Width < 2 && count > 0) fillBar.Width = 2;

            barGrid.Children.Add(bgBar);
            barGrid.Children.Add(fillBar);

            TextBlock countBlock = new TextBlock
            {
                Text = $"{count}",
                Foreground = WpfBrushes.White,
                FontWeight = FontWeights.Bold,
                VerticalAlignment = VerticalAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Right,
                FontSize = 12
            };

            Grid.SetColumn(labelBlock, 0);
            Grid.SetColumn(barGrid, 1);
            Grid.SetColumn(countBlock, 2);

            rowGrid.Children.Add(labelBlock);
            rowGrid.Children.Add(barGrid);
            rowGrid.Children.Add(countBlock);

            panel.Children.Add(rowGrid);
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
                ResetRightPanelToReady(); AccuracyResultTextBlock.Text = "정확도 테스트 대기 중";
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

        // [수정됨] 반환값 변경: (판정텍스트, 결함리스트)
        private async Task<(string DecisionText, List<DetectionResult> Defects)> RunFullPipelineAsync(string imagePath, BitmapImage bitmap)
        {
            DetectionCanvas.Children.Clear();
            DetectionResult topDetection;
            bool detectionSucceeded;
            List<DetectionResult> defectResults;

            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                // 1. 객체 탐지
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
                    string kname = _detectionTranslationMap.GetValueOrDefault(topDetection.ClassName, topDetection.ClassName);
                    DetectionResultTextBlock.Text = $"{kname} ({topDetection.Confidence * 100:F1}%)";
                }

                var cropBox = topDetection.Box;
                cropBox.Intersect(new ImageSharpRectangle(0, 0, originalImage.Width, originalImage.Height));
                if (cropBox.Width <= 0 || cropBox.Height <= 0) return ("오류", new List<DetectionResult>());

                // 2. 분석 실행
                var (koreanRipeness, englishRipeness, conf, allScores) = await RunClassificationAsync(originalImage, cropBox);
                defectResults = await RunDefectDetectionAsync(originalImage, cropBox);

                string varietyName = "";
                if (_varietySession != null)
                {
                    var (koreanVariety, _) = await RunVarietyClassificationAsync(originalImage, cropBox);
                    varietyName = koreanVariety;
                }

                // 3. 최종 판정
                var (decision, color, decisionColor) = GetFinalDecision(englishRipeness, defectResults, topDetection.Box);
                string estimatedWeight = EstimateWeightCategory(topDetection.Box);

                // UI 업데이트
                if (detectionSucceeded && !string.IsNullOrEmpty(varietyName)) DetectionResultTextBlock.Text = varietyName;
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

                // 이력 저장
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

                // 결과 반환
                return (decision, defectResults);
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

        // --- 추론 헬퍼 메서드들 ---
        private async Task<(string, float)> RunVarietyClassificationAsync(Image<Rgb24> img, ImageSharpRectangle box)
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

        private async Task<List<DetectionResult>> RunDefectDetectionAsync(Image<Rgb24> img, ImageSharpRectangle box)
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

        private async Task<(string, string, float, List<PredictionScore>)> RunClassificationAsync(Image<Rgb24> img, ImageSharpRectangle box)
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

        private DenseTensor<float> ImageToTensor(Image<Rgb24> img, int size)
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

        private (Image<Rgb24>, float, int, int) PreprocessDetectionImage(Image<Rgb24> original, int target)
        {
            float scale = Math.Min((float)target / original.Width, (float)target / original.Height);
            int nw = (int)(original.Width * scale), nh = (int)(original.Height * scale);
            var resized = original.Clone(x => x.Resize(nw, nh));
            int px = (target - nw) / 2, py = (target - nh) / 2;
            var final = new Image<Rgb24>(target, target, new Rgb24(114, 114, 114));
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

        private string EstimateWeightCategory(ImageSharpRectangle box)
        {
            long area = box.Width * box.Height;
            if (area < 50000) return "소 (150-300g)";
            if (area < 100000) return "중 (350-500g)";
            if (area < 150000) return "대 (500-650g)";
            return "특대 (600-750g)";
        }

        private void DrawBox(ImageSharpRectangle box, WpfBrush brush, double thick)
        {
            var r = new System.Windows.Shapes.Rectangle { Stroke = brush, StrokeThickness = thick, Width = box.Width, Height = box.Height };
            Canvas.SetLeft(r, box.X); Canvas.SetTop(r, box.Y);
            DetectionCanvas.Children.Add(r);
        }
    }
}
