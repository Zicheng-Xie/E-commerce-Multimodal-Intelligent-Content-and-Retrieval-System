// Read original local images and emit text observations in input order.
import Foundation
import Vision

struct Observation: Codable {
    let text: String
    let confidence: Float
}

struct ImageResult: Codable {
    let observations: [Observation]
    let error: String?
}

func recognize(_ path: String) -> ImageResult {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = false
    do {
        let url = URL(fileURLWithPath: path)
        try VNImageRequestHandler(url: url, options: [:]).perform([request])
        let lines = (request.results ?? []).compactMap { observation in
            observation.topCandidates(1).first.map {
                Observation(text: $0.string, confidence: $0.confidence)
            }
        }
        return ImageResult(observations: lines, error: nil)
    } catch {
        return ImageResult(observations: [], error: error.localizedDescription)
    }
}

let input = FileHandle.standardInput.readDataToEndOfFile()
let paths = try JSONDecoder().decode([String].self, from: input)
let results = paths.map(recognize)
let output = try JSONEncoder().encode(results)
FileHandle.standardOutput.write(output)
