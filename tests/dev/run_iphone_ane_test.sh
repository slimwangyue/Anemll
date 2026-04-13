#!/bin/bash
# iPhone ANE Test - RUN FROM Terminal.app (NOT from SSH/VS Code)
# Tests reduce_sum vs reduce_mean RMSNorm on iPhone A16 ANE
set -e
DEVICE="DAB3EAE5-B1DA-5B04-9EFC-3A9591592E93"
APP_ID="com.anemll.ANETestLoader"
MODELS="/Volumes/MySSD/tmp/ane_rmsnorm_test"

echo "=== 1. Build ==="
cd /Volumes/MySSD/Anemll/tests/dev/ANETestLoader
xcodebuild -project ANETestLoader.xcodeproj -scheme ANETestLoader \
  -destination "id=$DEVICE" -configuration Debug \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  DEVELOPMENT_TEAM=9KFM2UMQSM build 2>&1 | tail -3

APP=$(find /Volumes/MySSD/tmp/DerivedData/ANETestLoader* -name "ANETestLoader.app" -path "*/Debug-iphoneos/*" 2>/dev/null | head -1)
echo "Signed: $(codesign -v "$APP" 2>&1)"

echo "=== 2. Install ==="
xcrun devicectl device install app --device "$DEVICE" "$APP"

echo "=== 3. Push models ==="
for m in "$MODELS"/*.mlpackage; do
  echo "  $(basename "$m")"
  xcrun devicectl device copy to --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$APP_ID" \
    --source "$m" --destination "Documents/$(basename "$m")"
done

echo "=== 4. Launch & wait 60s ==="
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing "$APP_ID"
sleep 60

echo "=== 5. Results ==="
xcrun devicectl device copy from --device "$DEVICE" \
  --domain-type appDataContainer --domain-identifier "$APP_ID" \
  --source Documents/results.txt --destination /tmp/ane_results.txt
cat /tmp/ane_results.txt
