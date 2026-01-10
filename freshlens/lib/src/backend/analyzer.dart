import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:freshlens/src/backend/softmax.dart';
import 'package:http/http.dart' as http;
import 'dart:io';

class Analyzer {
  Future<AnalysisResult?> analyzeImage(String filePath) async {
    File imgFile = File(filePath);
    List<int> imageBytes = imgFile.readAsBytesSync();
    String base64Image = base64Encode(imageBytes);
    return await _sendToAzureModel(base64Image);
    //return Future.delayed(
    //    const Duration(seconds: 1), () => AnalysisResult(0.96, false));
  }

  Future<AnalysisResult> _sendToAzureModel(String base64Image) async {
    debugPrint("sending to azure model");
    String key = 'kHetYokhLMLwOGxTkbdreGyuvFEtJWlw';
    String url =
        'https://freshlens-1-1-wqtvpzsgdhprtkmwq.eastus2.inference.ml.azure.com/score';

    //String modelurl = 'https://freshlens-1-1-wqtvpzsgdhprtkmwq.eastus2.inference.ml.azure.com/score';
    //String apiKey = 'kHetYokhLMLwOGxTkbdreGyuvFEtJWlw';

    // request to json file to get the url and key
    try {
      final jsonfileResponse = await http.get(Uri.parse(
          'https://freshlensfiles267347.z20.web.core.windows.net/file15262351348295387.json'));

      if (jsonfileResponse.statusCode == 200) {
        Map<String, dynamic> jsonFileContent =
            jsonDecode(jsonfileResponse.body);
        url = jsonFileContent['u'];
        key = jsonFileContent['k'];
        debugPrint("loaded url from webserver");
      }
    } catch (e) {
      // just use the predefined url and key
      debugPrint("error loading url from webserver");
      key = 'kHetYokhLMLwOGxTkbdreGyuvFEtJWlw';
      url =
          'https://freshlens-1-1-wqtvpzsgdhprtkmwq.eastus2.inference.ml.azure.com/score';
    }

    var headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer $key'
    };

    var request = http.Request('POST', Uri.parse(url));
    request.body = json.encode({"data": base64Image});
    request.headers.addAll(headers);

    try {
      http.StreamedResponse response = await request.send();
      if (response.statusCode == 200) {
        var jsonResponseString = await response.stream.bytesToString();
        Map<String, dynamic> jsonResponse =
            jsonDecode(jsonDecode(jsonResponseString));
        debugPrint(jsonResponse.toString());
        int prediction = jsonResponse['prediction'] as int;
        List<double> scores = jsonResponse['output'][0].cast<double>();
        scores = softmax(scores);
        debugPrint(scores.toString());
        return AnalysisResult(scores[0], prediction == 1);
      } else {
        debugPrint(response.reasonPhrase);
        return AnalysisResult(0, false,
            error: response.reasonPhrase ?? "Error");
      }
    } catch (e) {
      debugPrint(e.toString());
      return AnalysisResult(0, false, error: e.toString());
    }
  }
}

class AnalysisResult {
  final double score;
  final bool moldDetected;
  String? error;
  AnalysisResult(this.score, this.moldDetected, {this.error});
}
