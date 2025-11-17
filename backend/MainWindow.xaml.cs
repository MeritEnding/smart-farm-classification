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
    // -----------------------------------------------------------------
    // [★ 추가] 분석 이력 저장을 위한 클래스
    // -----------------------------------------------------------------
    public class AnalysisHistoryItem
    {
        // UI 복원을 위한 데이터
        public BitmapImage Thumbnail { get; set; } // ListView 썸네일
        public BitmapImage FullImageSource { get; set; } // 원본 이미지
        public double OriginalImageWidth { get; set; }
        public double OriginalImageHeight { get; set; }

        // 그리기용 바운딩 박스 데이터
        public List<DetectionResult> MangoDetections { get; set; }
        public List<DetectionResult> DefectDetections { get; set; }

        // 텍스트 결과
        public string FileName { get; set; } // ListView 표시용
        public string DetectionResultText { get; set; }
        public string DetectedSizeText { get; set; }
        public string RipenessResultText { get; set; }
        public string ConfidenceText { get; set; }

        // 최종 결론
        public string FinalDecisionText { get; set; }
        public Brush FinalDecisionBackground { get; set; }
        public Brush FinalDecisionBrush { get; set; } // 이력 ListView 표시용

        // 하단 탭 상세 정보
        public IEnumerable<PredictionScore> AllRipenessScores { get; set; }
        public string DefectListText { get; set; }
        public Brush DefectListForeground { get; set; }
    }


    // 분류 모델 결과
    public class PredictionScore
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
    }

    // 탐지 모델 결과 (망고 탐지, 결함 탐지 공용)
    public class DetectionResult
    {
        public string ClassName { get; set; } = ""; // 예: "Mango" 또는 "scab" (로직을 위해 영어 원본 유지)
        public double Confidence { get; set; } // 예: 0.95
        public Rectangle Box { get; set; } // 이미지 내의 위치 (x, y, width, height)
    }

    public partial class MainWindow : Window
    {
        private InferenceSession? _classificationSession; // (best.onnx)
        private InferenceSession? _detectionSession;      // (detection.onnx - 망고 전체)
        private InferenceSession? _defectSession;         // (defect_detection.onnx - 망고 결함)

        // [★ 추가] 분석 이력 저장용 리스트
        private List<AnalysisHistoryItem> _analysisHistory = new List<AnalysisHistoryItem>();
        private bool _isHistoryLoading = false; // 재귀적 선택 방지 플래그

        // --- 분류 모델 (best.onnx) 설정 ---
        private readonly string[] _classificationClassNames = new string[]
        { "overripe", "breaking - stage","un-healthy", "ripe", "unripe", "half-riping-stage" };

        // 분류 모델 한글 변환 맵
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
        // 망고 탐지 클래스 한글 변환 맵
        private readonly Dictionary<string, string> _detectionTranslationMap = new Dictionary<string, string>
        {
            { "Mango", "망고" }
        };
        private const int DetectionInputSize = 640;


        // --- 결함 탐지 모델 (defect_detection.onnx) 설정 ---
        private readonly string[] _defectClassNames = new string[]
        {
        "brown-spot",         // data.yaml의 0번째 이름
            "black-spot",         // data.yaml의 1번째 이름
            "scab"                // data.yaml의 2번째 이름
        };
        // 결함 클래스 한글 변환 맵
        private readonly Dictionary<string, string> _defectTranslationMap = new Dictionary<string, string>
        {
            { "brown-spot", "갈색 반점" },
            { "black-spot", "검은 반점" },
            { "scab", "더뎅이병" }
        };
        private const int DefectInputSize = 640; // Colab 학습 시 640 사용


        public MainWindow()
        {
            InitializeComponent();
            LoadModelsAsync();

            // 환경/날씨 텍스트 (임시)
            FarmEnvTextBlock.Text = "온도: 28°C\n습도: 75%";
            WeatherTextBlock.Text = "맑음, 32°C\n바람: 3m/s";
            SeasonInfoTextBlock.Text = "수확기 (7월)\n품종: 애플망고";
        }

        /// <summary>
        /// 3개 모델을 비동기식으로 로드 (UI 차단 방지)
        /// </summary>
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

        /// <summary>
        /// 이미지 버튼 클릭 시
        /// </summary>
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
        // [★ 추가] 이력 불러오기 이벤트 핸들러
        // -----------------------------------------------------------------
        private void HistoryListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // 재귀 호출(무한 루프) 방지 및 유효성 검사
            if (_isHistoryLoading || e.AddedItems.Count == 0 || e.AddedItems[0] is not AnalysisHistoryItem selectedItem)
            {
                return;
            }

            _isHistoryLoading = true; // 불러오기 시작 플래그

            try
            {
                // 1. UI 상태 변경: 환영 패널 숨기고, 이미지 패널 표시
                WelcomePanel.Visibility = Visibility.Collapsed;
                ImagePreviewPanel.Visibility = Visibility.Visible;

                // 2. 이미지 및 캔버스 복원
                SourceImage.Source = selectedItem.FullImageSource;
                PreviewGrid.Width = selectedItem.OriginalImageWidth;
                PreviewGrid.Height = selectedItem.OriginalImageHeight;

                // 3. 바운딩 박스 복원
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

                // 4. 텍스트 결과 복원 (2x2 그리드)
                DetectionResultTextBlock.Text = selectedItem.DetectionResultText;
                DetectedSizeTextBlock.Text = selectedItem.DetectedSizeText;
                RipenessResultTextBlock.Text = selectedItem.RipenessResultText;
                ConfidenceTextBlock.Text = selectedItem.ConfidenceText;

                // 텍스트 색상 복원
                DetectionResultTextBlock.Foreground = Brushes.Orange; // #FFA500
                RipenessResultTextBlock.Foreground = Brushes.DodgerBlue; // #0091FF

                // 5. 최종 결론 복원
                FinalDecisionTextBlock.Text = selectedItem.FinalDecisionText;
                if (FinalDecisionTextBlock.Parent is Border decisionBorder)
                {
                    decisionBorder.Background = selectedItem.FinalDecisionBackground;
                }

                // 6. 하단 탭 (현재 분석) 복원
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
                _isHistoryLoading = false; // 불러오기 완료 플래그
            }
        }


        // -----------------------------------------------------------------
        // [이미지 처리 및 UI 리셋 헬퍼 함수]
        // -----------------------------------------------------------------

        /// <summary>
        /// (리팩토링) 실제 이미지 분석 및 UI 업데이트를 처리하는 핵심 함수
        /// </summary>
        private async Task ProcessImageAsync(string imagePath)
        {
            // [★ 추가] 새 분석 시작 시, 이력 리스트 선택 해제
            _isHistoryLoading = true; // 임시로 플래그 설정
            HistoryListView.SelectedIndex = -1;
            _isHistoryLoading = false; // 플래그 해제

            try
            {
                // --- 0. UI 초기화 (분석 시작) ---
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

                // UI 상태 변경: 환영 패널 숨기고, 이미지 패널 표시
                WelcomePanel.Visibility = Visibility.Collapsed;
                ImagePreviewPanel.Visibility = Visibility.Visible;

                // 분석 텍스트 설정
                DetectionResultTextBlock.Text = "탐지 중...";
                DetectedSizeTextBlock.Text = "분석 중...";
                RipenessResultTextBlock.Text = "분류 중...";
                ConfidenceTextBlock.Text = "분석 중...";
                FullResultsListView.ItemsSource = null;
                DefectResultsTextBlock.Text = "결함 탐지 중...";
                FinalDecisionTextBlock.Text = "판단 중...";
                if (FinalDecisionTextBlock.Parent is Border decisionBorder)
                {
                    decisionBorder.Background = Brushes.DarkSlateGray; // 판단 중 색상
                }

                // [★ 수정] RunFullPipelineAsync에 bitmap 객체 전달
                await RunFullPipelineAsync(imagePath, bitmap);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"처리 중 오류 발생: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                ResetToWelcomeState();
            }
        }

        /// <summary>
        /// 분석 실패 또는 초기화 시 환영 상태로 UI를 리셋합니다.
        /// </summary>
        private void ResetToWelcomeState()
        {
            Dispatcher.Invoke(() =>
            {
                // UI 상태 되돌리기
                WelcomePanel.Visibility = Visibility.Visible;
                ImagePreviewPanel.Visibility = Visibility.Collapsed;
                SourceImage.Source = null;

                // 캔버스 지우기
                DetectionCanvas.Children.Clear();
                PreviewGrid.Width = double.NaN;
                PreviewGrid.Height = double.NaN;

                // 오른쪽 패널 텍스트 리셋
                ResetRightPanelToReady();

                // [★ 추가] 이력 리스트 선택 해제
                _isHistoryLoading = true;
                HistoryListView.SelectedIndex = -1;
                _isHistoryLoading = false;
            });
        }

        /// <summary>
        /// 오른쪽 분석 패널을 '준비' 또는 '실패' 상태로 깔끔하게 초기화합니다.
        /// </summary>
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


        /// <summary>
        /// [★ 수정된 함수]
        /// 전체 파이프라인 (이력 저장을 위해 BitmapImage 매개변수 추가)
        /// </summary>
        private async Task RunFullPipelineAsync(string imagePath, BitmapImage bitmap)
        {
            DetectionCanvas.Children.Clear();

            DetectionResult topDetection;
            string detectionText;
            bool detectionSucceeded;
            List<DetectionResult> defectResults; // 이력 저장을 위해 상위 스코프로 이동
            string defectListText; // 이력 저장을 위해 상위 스코프로 이동
            Brush defectListForeground; // 이력 저장을 위해 상위 스코프로 이동

            using (var originalImage = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
            {
                var detectionResults = await RunDetectionAsync(imagePath);

                if (detectionResults == null || !detectionResults.Any())
                {
                    detectionText = "물체 탐지 실패 (전체 분석)";
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
                    // ... (이하 생략)
                    return;
                }

                var (koreanPredictedClass, englishPredictedClass, confidence, allScores) = await RunClassificationAsync(originalImage, cropBox);
                defectResults = await RunDefectDetectionAsync(originalImage, cropBox); // 상위 스코프 변수에 할당
                var (decision, color, decisionColor) = GetFinalDecision(englishPredictedClass, defectResults, topDetection.Box);

                string estimatedWeight = EstimateWeightCategory(topDetection.Box);

                // --- UI 업데이트 ---
                DetectionResultTextBlock.Text = detectionText;
                DetectedSizeTextBlock.Text = estimatedWeight;
                RipenessResultTextBlock.Text = $"{koreanPredictedClass}";
                ConfidenceTextBlock.Text = $"{confidence * 100:F2} %";
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

                // --- [★ 추가] 분석 완료 후 이력 저장 ---
                var historyItem = new AnalysisHistoryItem
                {
                    Thumbnail = bitmap, // 썸네일 (간단하게 원본 사용)
                    FullImageSource = bitmap,
                    OriginalImageWidth = bitmap.PixelWidth,
                    OriginalImageHeight = bitmap.PixelHeight,
                    FileName = System.IO.Path.GetFileName(imagePath), // 파일명

                    // 박스 데이터
                    MangoDetections = new List<DetectionResult> { topDetection },
                    DefectDetections = defectResults,

                    // 텍스트 데이터
                    DetectionResultText = detectionText,
                    DetectedSizeText = estimatedWeight,
                    RipenessResultText = koreanPredictedClass,
                    ConfidenceText = $"{confidence * 100:F2} %",

                    // 결론
                    FinalDecisionText = decision,
                    FinalDecisionBackground = decisionColor,
                    FinalDecisionBrush = (decisionColor == Brushes.DarkRed || decisionColor == Brushes.DarkOrange) ? Brushes.Tomato : Brushes.LightGreen,

                    // 하단 탭
                    AllRipenessScores = allScores,
                    DefectListText = defectListText,
                    DefectListForeground = defectListForeground
                };

                _analysisHistory.Insert(0, historyItem); // 새 항목을 맨 위에 추가

                // 이력 ListView 업데이트 (ItemsSource를 새로고침)
                HistoryListView.ItemsSource = null;
                HistoryListView.ItemsSource = _analysisHistory;
            }
        }


        // -----------------------------------------------------------------
        // [ 핵심 로직: 최종 판매 결정 ] (이전과 동일)
        // -----------------------------------------------------------------
        private (string Decision, Brush TextColor, Brush BackgroundColor) GetFinalDecision(string englishRipeness, List<DetectionResult> defects, Rectangle mangoBox)
        {
            Brush PASS_COLOR = new SolidColorBrush(System.Windows.Media.Color.FromRgb(0x2E, 0xCC, 0x71));
            Brush REJECT_COLOR = Brushes.DarkRed;
            Brush CONDITIONAL_COLOR = Brushes.DarkOrange;
            Brush HOLD_COLOR = Brushes.DarkSlateGray;
            Brush TEXT_COLOR = Brushes.White;

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

            bool hasScab = defects.Any(d => d.ClassName == "scab");
            bool hasBrownSpot = defects.Any(d => d.ClassName == "brown-spot");
            bool hasBlackSpot = defects.Any(d => d.ClassName == "black-spot");
            bool hasOtherDefects = defects.Any(d => d.ClassName != "black-spot");

            if (englishRipeness == "overripe") return ("폐기 (과숙)", TEXT_COLOR, REJECT_COLOR);
            if (englishRipeness == "un-healthy") return ("폐기 (비정상/병함)", TEXT_COLOR, REJECT_COLOR);
            if (defectRatio > 0.10) return ($"폐기 (결함 면적 {defectRatio:P0} > 10%)", TEXT_COLOR, REJECT_COLOR);
            if (hasScab && defectRatio > 0.05) return ($"폐기 ({_defectTranslationMap["scab"]} 결함 5% 초과)", TEXT_COLOR, REJECT_COLOR);
            if (hasBrownSpot && defectRatio > 0.05) return ($"폐기 ({_defectTranslationMap["brown-spot"]} 5% 초과)", TEXT_COLOR, REJECT_COLOR);

            bool passRipeness = (englishRipeness == "half-riping-stage" || englishRipeness == "ripe");
            bool passDefectRatio = (defectRatio <= 0.05);
            bool passDefectType = !hasOtherDefects;
            if (passRipeness && passDefectRatio && passDefectType) return ("정상 판매 가능", TEXT_COLOR, PASS_COLOR);

            bool condRipeness = (englishRipeness == "breaking - stage" || englishRipeness == "ripe");
            bool condDefectRatio = (defectRatio > 0.05 && defectRatio <= 0.10);
            bool condDefectType = !hasScab;
            if (condRipeness && condDefectRatio && condDefectType) return ("저가 판매 / 즉시 유통", TEXT_COLOR, CONDITIONAL_COLOR);

            if (englishRipeness == "unripe") return ("판단 보류 (미숙)", TEXT_COLOR, HOLD_COLOR);

            return ("판단 보류 (규칙 외)", TEXT_COLOR, HOLD_COLOR);
        }

        // -----------------------------------------------------------------
        // [ 이하 모델 추론 및 헬퍼 함수 ] (이전과 동일)
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
                                    ClassName = _defectClassNames[maxClassId], // "scab" (영어)
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
                                    ClassName = _detectionClassNames[maxClassId], // "Mango" (영어)
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
                        var probabilities = Softmax(output.ToArray());
                        var allScores = new List<PredictionScore>();
                        for (int i = 0; i < probabilities.Length; i++)
                        {
                            string englishName = _classificationClassNames[i];
                            string koreanName = _translationMap[englishName];
                            allScores.Add(new PredictionScore { ClassName = koreanName, Confidence = probabilities[i] });
                        }
                        float maxConfidence = probabilities.Max();
                        int maxIndex = Array.IndexOf(probabilities, maxConfidence);
                        string englishTopClass = _classificationClassNames[maxIndex];
                        string koreanTopClass = _translationMap[englishTopClass];
                        return (koreanTopClass, englishTopClass, maxConfidence, allScores);
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
