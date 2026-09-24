# Part 3 — Problem-Solving Answers

## 1. The classifier is wrong 30% of the time. What do you do, and how do you decide whether it is good enough?

If the classifier is wrong 30% of the time, I would first try to understand **where those errors are coming from** rather than immediately trying to improve the overall accuracy.

I would start by looking at the confusion matrix and per-class precision and recall. An overall 70% accuracy could hide the fact that some classes are classified very well while another important class is performing poorly.

Then I would inspect some of the incorrectly classified tiles. I would check whether the images are genuinely ambiguous, whether the labels are correct, and whether there is some systematic confusion between particular classes.

I would also verify the complete input pipeline: whether the image is being decoded correctly, converted to RGB, resized to the expected size, and normalized in the same way as during training. This helps me determine whether the problem is actually with the model or with the data and preprocessing pipeline.

After that, if the errors are mainly a model limitation, I would consider improvements such as adding more representative training data, fine-tuning more of the backbone, using augmentation, increasing the input resolution if appropriate, or trying another model.

For deciding whether it is "good enough", I would not use a fixed rule such as "70% accuracy is good" or "90% accuracy is good". I would first understand **what the predictions are going to be used for and what happens when a prediction is wrong**.

For example, if a wrong classification is only used as an initial suggestion to an analyst, 70% might be usable with human review. If an incorrect classification directly triggers an important downstream decision, the required accuracy and class-level performance would be much higher.

I would therefore define an acceptance criterion based on the actual use case, including overall accuracy, per-class performance, and the cost of different types of errors. I would also validate any confidence threshold before relying on it for automatically sending predictions for review.

In our actual implementation, the held-out evaluation accuracy is 87.14% (183/210), but I would use that as an evaluation result rather than automatically calling it "good enough." The same reasoning about the intended use and error cost still applies.

---

## 2. The service runs offline and nobody watches it. A month after deployment, how would you know it is still working correctly?

If nobody is watching the service, I would not rely on someone manually checking it. I would build checks into the system so that failures can be detected automatically.

First, I would have a health or readiness check that verifies that the service is running, the model is loaded, and the database is accessible.

I would also record structured logs and basic metrics such as:

- number of requests processed
- successful and failed requests
- inference failures
- storage failures
- request and inference latency
- database/disk health

For an offline system, I would also use a **known canary image**. Periodically, the service could run the same known input through the complete classification pipeline and check that the result is within an expected range. This would help detect problems with the model, preprocessing, or runtime environment even when normal users are not reporting failures.

I would also persist the model version with every result, as we already do in this implementation. That makes it possible to determine which model produced a particular result if the model is updated later.

Finally, since nobody is watching the service directly, the health information needs to be stored or surfaced somewhere that can be checked later, and alerts should be generated when possible.

The current take-home implementation does not include full production monitoring or alerting. For this thin slice, I focused on the classification and persistence path. For an unattended deployment, I would add health checks, monitoring, and periodic canary classification.

---

## 3. Tiles are coming in fine, but the stored results look wrong. Walk me through how you would find the cause, in order.

I would debug this from the outside of the system toward the model, rather than immediately changing the model.

### 1. Identify the affected results

First, I would identify the result IDs and tile IDs that look wrong.

I would check whether the problem affects:

- one tile
- a particular class
- a particular time period
- all incoming tiles

This tells me whether the problem is isolated or systematic.

### 2. Inspect the stored result

I would inspect the stored prediction, confidence, status, timestamp, and model version.

This tells me whether the problem is actually in the prediction itself or whether the prediction was correct but something went wrong while storing or returning the result.

### 3. Check logs around that request

I would check the application logs for errors or exceptions around the time that result was created.

I would specifically look for inference failures, storage failures, retries, or unexpected request behavior.

### 4. Re-run the exact same tile

I would take the exact input tile and run it again through the same preprocessing and model artifact.

If the direct re-run gives the expected result but the stored API result is wrong, I would focus on the API or persistence path.

If the direct re-run is also wrong, I would continue toward the input, preprocessing, or model.

### 5. Check the input

I would verify the image itself:

- dimensions
- channels
- format
- whether it can be decoded correctly

I would make sure the actual input matches what the model expects.

### 6. Check preprocessing

I would verify that the same preprocessing is being used:

- RGB conversion
- resize to 64×64
- scaling
- ImageNet normalization
- resulting tensor/array shape

A preprocessing mismatch can make a correct model appear to be producing bad results.

### 7. Check the model artifact and version

I would verify that the expected local model artifact is loaded and that the stored `model_version` matches the artifact that was supposed to produce the result.

This is particularly important if models have been updated.

### 8. Inspect the model output

I would inspect the model's predicted class, confidence, and, during a reproduction run, its raw output/logits.

If the model is confidently predicting the wrong class, that points more toward an input/model/data issue. If the prediction is close between classes, the input may simply be ambiguous.

### 9. Check persistence

If the model produced the correct result but the stored result is wrong, I would inspect the database insert and read path.

I would check things such as enum-to-string conversion, serialization, and deserialization.

### 10. Compare with evaluation behavior

I would check whether similar tiles or the same class show similar behavior in the held-out evaluation set.

This helps determine whether the issue is an existing model limitation or a new regression.

### 11. Isolate the root cause

At this point I would classify the problem as primarily an:

- input problem
- preprocessing problem
- model problem
- API problem
- persistence problem

Then I would fix the relevant layer rather than making unrelated changes.

### 12. Add a regression test

Before or alongside the fix, I would reproduce the problem in a test so that once it is fixed, the same problem cannot silently come back.

---

## 4. What is the weakest part of your design, and what would break it first?

The weakest part of my design is the **single-process architecture with SQLite storage**.

I chose it deliberately because the assignment asks for a small offline working slice, and it avoids introducing unnecessary infrastructure. But it also creates a clear scalability limitation.

The first thing I would expect to break under a significantly higher workload is **throughput and concurrency**.

The classifier runs inside the same process as the API, and the results are stored in SQLite. SQLite is convenient for a local offline application, but concurrent writes are more limited than they would be with a database designed for higher write concurrency.

As the number of simultaneous classification requests increases, I would expect database contention and request latency to become problems. Since the whole service is also a single process, I cannot simply scale the application horizontally while keeping the same local database setup.

For the offline, low-volume use case of this take-home, I think this is a reasonable trade-off because it keeps the system simple and easy to operate.

Other limitations I would be aware of are the lack of production monitoring, uncalibrated confidence scores, no raw-image retention, and the limited analyst query API.

If this system had to move toward a higher-volume production deployment, I would revisit the database and process architecture first, while also adding proper health monitoring and operational observability.