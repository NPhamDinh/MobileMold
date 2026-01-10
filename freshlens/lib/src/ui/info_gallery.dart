import 'package:flutter/material.dart';
import 'package:introduction_screen/introduction_screen.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'main_screen.dart';
import 'theme/constants.dart';

class InfoGallery extends StatefulWidget {
  static const routeName = '/info';
  const InfoGallery({super.key});

  @override
  State<InfoGallery> createState() => _InfoGalleryState();
}

class _InfoGalleryState extends State<InfoGallery> {
  final _introKey = GlobalKey<IntroductionScreenState>();

  @override
  Widget build(BuildContext context) {
    const paddedDecoration = PageDecoration(
      bodyPadding: EdgeInsets.all(kDefaultPadding),
      imagePadding: EdgeInsets.symmetric(
          horizontal: kDefaultPadding, vertical: 2 * kDefaultPadding),
      titlePadding: EdgeInsets.symmetric(horizontal: kDefaultPadding),
    );
    return Container(
      color: Theme.of(context).colorScheme.surface,
      child: SafeArea(
        child: IntroductionScreen(
          key: _introKey,
          pages: [
            PageViewModel(
              decoration: paddedDecoration,
              titleWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "WELCOME!",
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
              ),
              image: Image.asset("assets/logos/Logo V2.png"),
              bodyWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "FreshLens detects mold on your food using AI.",
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
              ),
            ),
            PageViewModel(
              decoration: paddedDecoration,
              titleWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "HOW IT WORKS",
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
              ),
              image: Image.asset(
                "assets/images/Brombeeren_square.jpg",
              ),
              bodyWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "FreshLens uses a smartphone clip-on microscope. Take a picture of your food using the clip-on microscope and FreshLens will analyze it for mold.",
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
              ),
            ),
            PageViewModel(
              decoration: paddedDecoration,
              titleWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "MINIMUM VIABLE PRODUCT",
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
              ),
              image: Image.asset(
                "assets/images/Step2.jpg",
              ),
              bodyWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "Attention. This is a minimum viable product for demonstration purposes. Consumption of food is at your own risk.",
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
              ),
            ),
            PageViewModel(
              decoration: paddedDecoration,
              titleWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "MICROSCOPE",
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
              ),
              bodyWidget: Container(
                alignment: Alignment.centerLeft,
                child: Column(
                  children: [
                    Text(
                      "FreshLens needs a smartphone clip-on microscope to work. It needs approximately 10x physical magnification.",
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                    const SizedBox(
                      height: kDefaultPadding,
                    ),
                    OutlinedButton(
                        onPressed: () {
                          _introKey.currentState?.next();
                        },
                        child: Container(
                            alignment: Alignment.center,
                            width: double.infinity,
                            child: const Text("I HAVE A SUITABLE MICROSCOPE"))),
                    const SizedBox(
                      height: 2 * kDefaultPadding,
                    ),
                    Text(
                      "If you don't own a microscope, you can buy one at our partner shops with a discount!",
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                    const SizedBox(
                      height: kDefaultPadding,
                    ),
                    OutlinedButton(
                        onPressed: () {
                          showDialog(
                              context: context,
                              builder: ((context) => AlertDialog(
                                    title: const Text("Buy microscope"),
                                    content: const Text(
                                        "This functionality does not exist in the minimum viable product."),
                                    actions: [
                                      TextButton(
                                          onPressed: () {
                                            Navigator.of(context).pop();
                                          },
                                          child: const Text("OK")),
                                    ],
                                  )));
                        },
                        child: Container(
                            alignment: Alignment.center,
                            width: double.infinity,
                            child: const Text("OUR RECOMMENDATIONS"))),
                  ],
                ),
              ),
            ),
            PageViewModel(
              decoration: paddedDecoration,
              titleWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "CHOOSING THE RIGHT CAMERA",
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
              ),
              image: Image.asset("assets/images/Front-back.jpg"),
              bodyWidget: Container(
                alignment: Alignment.centerLeft,
                child: Text(
                  "We recommend placing the clip-on microscope lens over your main back camera. Phones with multiple back cameras can cause problems. If you can't mount your microscope on the back or experience problems, mount it on the front camera.",
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
              ),
            ),
          ],
          showDoneButton: true,
          showBackButton: true,
          back: Text(
            "Back",
            style: Theme.of(context).textTheme.displaySmall,
          ),
          showNextButton: true,
          next: Text(
            "Next",
            style: Theme.of(context).textTheme.displaySmall,
          ),
          done: Text(
            "Let's go!",
            style: Theme.of(context).textTheme.displaySmall,
          ),
          onDone: () async {
            final prefs = await SharedPreferences.getInstance();
            await prefs.setBool('hasSeenInfoGallery', true);
            if (mounted) {
              Navigator.of(context).pushNamedAndRemoveUntil(
                  MainScreen.routeName, (route) => false);
            }
          },
        ),
      ),
    );
  }
}
