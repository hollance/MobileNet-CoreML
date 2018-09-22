import UIKit
import Vision
import AVFoundation
import CoreMedia

class ViewController: UIViewController {
  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var predictionLabel: UILabel!
  @IBOutlet weak var timeLabel: UILabel!

  // true: use Vision to drive Core ML, false: use plain Core ML
  let useVision = false

  // Vision automatically resizes the images, but regular Core ML doesn't.
  let inputWidth = 224
  let inputHeight = 224

  // How many predictions we can do concurrently.
  static let maxInflightBuffers = 3

  let model = MobileNet()

  var videoCapture: VideoCapture!
  var requests = [VNCoreMLRequest]()
  var startTimes: [CFTimeInterval] = []

  var framesDone = 0
  var frameCapturingStartTime = CACurrentMediaTime()

  var inflightBuffer = 0
  let semaphore = DispatchSemaphore(value: ViewController.maxInflightBuffers)

  override func viewDidLoad() {
    super.viewDidLoad()

    predictionLabel.text = ""
    timeLabel.text = ""

    setUpVision()
    setUpCamera()

    frameCapturingStartTime = CACurrentMediaTime()
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  // MARK: - Initialization

  func setUpVision() {
    guard let visionModel = try? VNCoreMLModel(for: model.model) else {
      print("Error: could not create Vision model")
      return
    }

    for _ in 0..<ViewController.maxInflightBuffers {
      let request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
      request.imageCropAndScaleOption = .centerCrop
      requests.append(request)
    }
  }

  func setUpCamera() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self
    videoCapture.desiredFrameRate = 240
    videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.hd1280x720) { success in
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.resizePreviewLayer()
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  // MARK: - UI stuff

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = videoPreview.bounds
  }

  // MARK: - Doing inference

  typealias Prediction = (String, Double)

  func predict(pixelBuffer: CVPixelBuffer) {
    // Measure how long it takes to predict a single video frame.
    let startTime = CACurrentMediaTime()

    // Resize the input using vImage.
    if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
                                                  width: inputWidth,
                                                  height: inputHeight) {
      // Give the resized input to our model.
      if let prediction = try? model.prediction(data: resizedPixelBuffer) {
        let top5 = top(5, prediction.prob)
        let elapsed = CACurrentMediaTime() - startTime

        DispatchQueue.main.async {
          self.show(results: top5, latency: elapsed)
        }
      } else {
        print("BOGUS")
      }
    }
    self.semaphore.signal()
  }

  func predictUsingVision(pixelBuffer: CVPixelBuffer) {
    // Measure how long it takes to predict a single video frame. Note that
    // predict() can be called on the next frame while the previous one is
    // still being processed. Hence the need to queue up the start times.
    startTimes.append(CACurrentMediaTime())

    // Vision will automatically resize the input image.
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
    let request = requests[inflightBuffer]

    // For better throughput, we want to schedule multiple Vision requests
    // in parllel. These need to be separate instances, and inflightBuffer
    // is the index of the current request object to use.
    inflightBuffer += 1
    if inflightBuffer >= ViewController.maxInflightBuffers {
      inflightBuffer = 0
    }

    // Because perform() will block until after the request completes, we
    // run it on a concurrent background queue, so that the next frame can
    // be scheduled in parallel with this one.
    DispatchQueue.global().async {
      try? handler.perform([request])
    }
  }

  func visionRequestDidComplete(request: VNRequest, error: Error?) {
    if let observations = request.results as? [VNClassificationObservation] {

      // The observations appear to be sorted by confidence already, so we
      // take the top 5 and map them to an array of (String, Double) tuples.
      let top5 = observations.prefix(through: 4)
                             .map { ($0.identifier, Double($0.confidence)) }

      let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)

      DispatchQueue.main.async {
        self.show(results: top5, latency: elapsed)
      }
    }

    self.semaphore.signal()
  }

  func show(results: [Prediction], latency: CFTimeInterval) {
    var s: [String] = []
    for (i, pred) in results.enumerated() {
      s.append(String(format: "%d: %@ (%3.2f%%)", i + 1, pred.0, pred.1 * 100))
    }
    predictionLabel.text = s.joined(separator: "\n\n")

    let fps = self.measureFPS()
    timeLabel.text = String(format: "%.2f FPS (latency %.5f seconds)", fps, latency)
  }

  func measureFPS() -> Double {
    // Measure how many frames were actually delivered per second.
    framesDone += 1
    let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
    let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
    if frameCapturingElapsed > 1 {
      framesDone = 0
      frameCapturingStartTime = CACurrentMediaTime()
    }
    return currentFPSDelivered
  }
}

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
    if let pixelBuffer = pixelBuffer {
      // The semaphore will block the capture queue and drop frames when
      // Core ML can't keep up with the camera.
      semaphore.wait()

      if useVision {
        // This method should always be called from the same thread!
        // Ain't nobody likes race conditions and crashes.
        self.predictUsingVision(pixelBuffer: pixelBuffer)
      } else {
        // For better throughput, perform the prediction on a concurrent
        // background queue instead of on the serial VideoCapture queue.
        DispatchQueue.global().async {
          self.predict(pixelBuffer: pixelBuffer)
        }
      }
    }
  }
}

public func top(_ k: Int, _ prob: [String: Double]) -> [(String, Double)] {
  return Array(prob.map { x in (x.key, x.value) }
                   .sorted(by: { a, b -> Bool in a.1 > b.1 })
                   .prefix(min(k, prob.count)))
}
