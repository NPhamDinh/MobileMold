import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:freshlens/src/backend/analyzer.dart';
import 'package:freshlens/src/ui/theme/constants.dart';

class PreviewPage extends StatefulWidget {
  static const routeName = '/result';
  final String imagePath;
  const PreviewPage({Key? key, required this.imagePath}) : super(key: key);

  @override
  State<PreviewPage> createState() => _PreviewPageState();
}

class _PreviewPageState extends State<PreviewPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Align(
            alignment: Alignment.topCenter,
            child: AspectRatio(
                aspectRatio: 1.0,
                child: Image.file(
                  File(widget.imagePath),
                  fit: BoxFit.cover,
                )),
          ),
          SizedBox(
            height: double.infinity,
            child: Align(
              alignment: Alignment.bottomCenter,
              child: ResultBottomPanel(
                imagePath: widget.imagePath,
              ),
            ),
          )
        ],
      ),
    );
  }
}

class ResultBottomPanel extends StatelessWidget {
  final String imagePath;
  const ResultBottomPanel({super.key, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    var size = MediaQuery.of(context).size;
    return Container(
      height: max(size.height - size.width + kBorderRadius, 400),
      padding: EdgeInsets.only(
        top: kDefaultPadding,
        bottom: MediaQuery.of(context).padding.bottom + kDefaultPadding,
        left: MediaQuery.of(context).padding.left + kDefaultPadding,
        right: MediaQuery.of(context).padding.right + kDefaultPadding,
      ),
      width: double.infinity,
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(kBorderRadius),
          topRight: Radius.circular(kBorderRadius),
        ),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(
          "RESULT",
          style: Theme.of(context).textTheme.headlineLarge,
        ),
        Expanded(child: ResultText(imagePath: imagePath)),
        OutlinedButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Container(
            alignment: Alignment.center,
            width: double.infinity,
            child: const Text("BACK TO CAMERA"),
          ),
        ),
      ]),
    );
  }
}

class ResultText extends StatefulWidget {
  final String imagePath;
  const ResultText({
    super.key,
    required this.imagePath,
  });

  @override
  State<ResultText> createState() => _ResultTextState();
}

class _ResultTextState extends State<ResultText> {
  AnalysisResult? result;
  late Analyzer analyzer;

  @override
  void initState() {
    analyzer = Analyzer();
    analyzer.analyzeImage(widget.imagePath).then((analyisResult) {
      if (mounted) {
        setState(() {
          result = analyisResult;
        });
      }
    });
    super.initState();
  }

  Color? _getTextColor(BuildContext context, double score) {
    /// sample color from linear gradient. 0.0 is red, 0.5 is yellow, 1.0 is green
    final green = Theme.of(context).colorScheme.tertiary;
    const red = Colors.red;
    const yellow = Colors.orange;
    return score < 0.5
        ? Color.lerp(red, yellow, score * 2)
        : Color.lerp(yellow, green, (score - 0.5) * 2);
  }

  @override
  Widget build(BuildContext context) {
    return result == null
        ? const Center(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CircularProgressIndicator(),
                SizedBox(
                  height: kDefaultPadding,
                ),
                Text("Analyzing image. Please wait a few seconds.")
              ],
            ),
          )
        : result!.error != null
            ? Center(
                child: Text(
                  "Failed to fetch prediction. Ensure you have an internet connection and try again.",
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              )
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(
                    height: kDefaultPadding,
                  ),
                  Text(
                    result!.moldDetected
                        ? "Mold detected."
                        : "No mold detected.",
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const Spacer(),
                  Text(
                    "${(result!.score * 100).toInt()} %",
                    style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                        color: _getTextColor(context, result!.score),
                        fontSize: 60),
                  ),
                  Text(
                    "Safety score",
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                  const Spacer(),
                ],
              );
  }
}
