import UIKit

class ViewController: UIViewController {
  @IBOutlet weak var imageView: UIImageView!
  @IBOutlet weak var predictionLabel: UILabel!

  let model = MobileNet()

  override func viewDidLoad() {
    super.viewDidLoad()

    let image = UIImage(named: "cat224x224")!
    imageView.image = image
    predict(image: image)
  }

  func predict(image: UIImage) {
    if let prediction = try? model.prediction(data: image.pixelBuffer()) {
      let top5 = top(5, prediction.prob)

      var s: [String] = []
      for (i, pred) in top5.enumerated() {
        s.append(String(format: "%d: %@ (%3.2f%%)", i + 1, pred.0, pred.1 * 100))
      }
      predictionLabel.text = s.joined(separator: "\n\n")
    }
  }

  typealias Prediction = (String, Double)

  public func top(_ k: Int, _ prob: [String: Double]) -> [Prediction] {
    precondition(k <= prob.count)

    return Array(prob.map { x in (x.key, x.value) }
                     .sorted(by: { a, b -> Bool in a.1 > b.1 })
                     .prefix(through: k - 1))
  }
}
