// fs-recorder — Flight School's window recorder (Milestone 4).
//
// Flight School starts this as a sidecar. It watches ONE window through
// ScreenCaptureKit and listens (listen-only, never blocking or changing
// anything) for key presses and clicks. For every input it keeps the frame
// from just BEFORE it — what the person was looking at when they decided, not
// what the input did — writes that frame into the session folder, and prints
// one JSON line. Everything about what an input MEANS (actions, names, rows)
// is decided in Rust (recorder.rs); this file only sees and reports.
//
// Why a Swift helper: on macOS 15 and later the CoreGraphics capture calls
// (CGWindowListCreateImage, CGDisplayCreateImage) are marked obsolete and
// only ScreenCaptureKit is supported. ScreenCaptureKit is Swift/Objective-C.
//
// Subcommands (all print JSON lines on stdout):
//   permission            {"screen":bool,"input":bool}   — never prompts
//   request               asks macOS for both, then prints the same line
//   windows               the windows that could be recorded
//   record <options…>     records until stdin says "stop" or closes
//   selftest <dir>        the frame-choosing and stacking code on drawn frames
//   hash                  a fingerprint for each image path read from stdin
//   synthetic <options…>  `record`, with drawn frames and scripted inputs —
//                         the same writer and the same lines, no permission
//
// Options for record/synthetic:
//   --window ID  --out DIR  --keys 0|1  --clicks 0|1
//   --before-ms 80  --idle-s 3  --stack 1  --stack-gap-ms 250  --max-side 768

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import AppKit
import Carbon.HIToolbox   // IsSecureEventInputEnabled
import ScreenCaptureKit
import CoreMedia
import CoreImage

// MARK: - output

let out = FileHandle.standardOutput
let outLock = NSLock()
func emit(_ obj: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else { return }
    outLock.lock(); defer { outLock.unlock() }
    out.write(data)
    out.write("\n".data(using: .utf8)!)
}

func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000 }

// MARK: - options

struct Options {
    var window: UInt32 = 0
    var out = ""
    var keys = true
    var clicks = true
    var beforeMs = 80.0
    var idleS = 3.0
    var stack = 1
    var stackGapMs = 250.0
    var maxSide = 768
    var crop: Crop?

    init(_ args: ArraySlice<String>) {
        var it = args.makeIterator()
        while let k = it.next() {
            let v = it.next() ?? ""
            switch k {
            case "--window": window = UInt32(v) ?? 0
            case "--out": out = v
            case "--keys": keys = v != "0"
            case "--clicks": clicks = v != "0"
            case "--before-ms": beforeMs = Double(v) ?? 80
            case "--idle-s": idleS = Double(v) ?? 3
            case "--stack": stack = max(1, min(4, Int(v) ?? 1))
            case "--stack-gap-ms": stackGapMs = Double(v) ?? 250
            case "--max-side": maxSide = max(64, Int(v) ?? 1280)
            case "--crop": crop = Crop(v)
            default: break
            }
        }
    }
}

// MARK: - the part of the window kept

/// The part of the window a recording keeps, as fractions of the window from
/// its top left — "x,y,w,h" — so it is the same part at any window size.
/// Pigeon's eye crops the same way, so a Pigeon sees what it learned from.
struct Crop {
    let x, y, w, h: Double
    init?(_ s: String) {
        let v = s.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard v.count == 4, v[2] > 0, v[3] > 0 else { return nil }
        x = min(max(v[0], 0), 1); y = min(max(v[1], 0), 1)
        w = min(v[2], 1 - x); h = min(v[3], 1 - y)
        guard w > 0.01, h > 0.01 else { return nil }
    }
    /// The kept part of a picture `W` × `H` pixels, top-left origin.
    func rect(_ W: Int, _ H: Int) -> CGRect {
        CGRect(x: (x * Double(W)).rounded(), y: (y * Double(H)).rounded(),
               width: (w * Double(W)).rounded(), height: (h * Double(H)).rounded())
            .intersection(CGRect(x: 0, y: 0, width: W, height: H))
    }
}

/// A frame cut to the kept part (CGImage's own pixels run from the top left),
/// or nil when the kept part rounds to under a pixel. Never the whole frame
/// instead: a Pigeon shown the whole window would be shown a picture it never
/// learned from (format §13.2, D66). Pigeon's Eye.swift does the same.
func cropped(_ img: CGImage, _ c: Crop?) -> CGImage? {
    guard let c else { return img }
    let r = c.rect(img.width, img.height)
    guard r.width >= 1, r.height >= 1 else { return nil }
    return img.cropping(to: r)
}

/// How big to capture a window of `w` × `h` pixels so that the kept part's
/// longer side is at most `maxSide` — never larger than the window really is.
func captureSize(_ w: Double, _ h: Double, maxSide: Int, crop: Crop?) -> (Double, Double) {
    let kw = w * (crop?.w ?? 1), kh = h * (crop?.h ?? 1)
    let k = min(1, Double(maxSide) / max(kw, kh))
    return ((w * k).rounded(), (h * k).rounded())
}

// MARK: - frames: the ring, the choice, the stack, the file

/// The last second or two of the window, newest last. Frames are copies
/// (CGImage), so holding them never starves the capture's own buffer pool.
final class Ring {
    private var frames: [(t: Double, image: CGImage)] = []
    private let lock = NSLock()
    let keepMs: Double
    init(keepMs: Double) { self.keepMs = keepMs }

    func add(_ image: CGImage, at t: Double) {
        lock.lock(); defer { lock.unlock() }
        frames.append((t, image))
        while let first = frames.first, t - first.t > keepMs, frames.count > 1 { frames.removeFirst() }
    }

    var latest: (t: Double, image: CGImage)? { lock.lock(); defer { lock.unlock() }; return frames.last }

    /// The frame the person was looking at `beforeMs` before the input: the
    /// newest one at or before that moment. If the window has not changed for
    /// a while there is no newer frame, and the newest one IS what was on
    /// screen. If every frame is newer (the input came in the first instant),
    /// the oldest is the closest there is, and its age says so.
    func before(_ t: Double, beforeMs: Double) -> (t: Double, image: CGImage)? {
        lock.lock(); defer { lock.unlock() }
        let cut = t - beforeMs
        return frames.last(where: { $0.t <= cut }) ?? frames.first
    }
}

/// `n` frames ending at the decision frame, `gapMs` apart, side by side,
/// oldest on the left, all at the decision frame's height. With n = 1 it is
/// the decision frame itself.
func stacked(_ ring: Ring, at t: Double, beforeMs: Double, n: Int, gapMs: Double) -> CGImage? {
    guard let last = ring.before(t, beforeMs: beforeMs) else { return nil }
    if n <= 1 { return last.image }
    var parts: [CGImage] = []
    for i in stride(from: n - 1, through: 1, by: -1) {
        if let f = ring.before(last.t - Double(i) * gapMs, beforeMs: 0) { parts.append(f.image) }
    }
    parts.append(last.image)
    let h = last.image.height
    let widths = parts.map { Int((Double($0.width) * Double(h) / Double($0.height)).rounded()) }
    let gutter = 4
    let w = widths.reduce(0, +) + gutter * (parts.count - 1)
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    ctx.setFillColor(CGColor(gray: 0, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    var x = 0
    for (p, pw) in zip(parts, widths) {
        ctx.draw(p, in: CGRect(x: x, y: 0, width: pw, height: h))
        x += pw + gutter
    }
    return ctx.makeImage()
}

func writeJPEG(_ image: CGImage, to url: URL) -> Bool {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { return false }
    CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
    return CGImageDestinationFinalize(dest)
}

// MARK: - key names

/// A key's name, the same on every keyboard layout for the keys that have no
/// letter, and the letter itself (upper case) for those that do. A modifier
/// held with it is part of the name: Cmd+S is not S.
func keyName(code: Int64, chars: String, flags: CGEventFlags) -> String? {
    let named: [Int64: String] = [
        36: "Return", 76: "Enter", 48: "Tab", 49: "Space", 51: "Backspace", 117: "Delete", 53: "Escape",
        123: "Left", 124: "Right", 125: "Down", 126: "Up", 115: "Home", 119: "End", 116: "PageUp", 121: "PageDown",
        122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5", 97: "F6", 98: "F7", 100: "F8", 101: "F9",
        109: "F10", 103: "F11", 111: "F12",
    ]
    // Modifier keys on their own are not inputs.
    if [54, 55, 56, 57, 58, 59, 60, 61, 62, 63].contains(code) { return nil }
    var base = named[code]
    if base == nil {
        let c = chars.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = c.first, !c.isEmpty, first.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }) else {
            return "Key\(code)"
        }
        base = String(first).uppercased()
    }
    var mods: [String] = []
    if flags.contains(.maskControl) { mods.append("Ctrl") }
    if flags.contains(.maskAlternate) { mods.append("Option") }
    if flags.contains(.maskCommand) { mods.append("Cmd") }
    return (mods + [base!]).joined(separator: "+")
}

// MARK: - the session: inputs in, frames out, lines printed

final class Session {
    let opts: Options
    /// Whether macOS had secure input on (a password field has focus) the last
    /// time a key arrived. Keys are never kept while it is on, and each change
    /// is reported once so the app can say so.
    var secureWas = false
    let ring: Ring
    let dir: URL
    var n = 0
    var lastInput = nowMs()
    var frameW = 0, frameH = 0
    /// The window in global screen points, top-left origin (as CGEvent reports).
    var bounds = CGRect.zero
    var ownerPid: pid_t = 0

    init(_ opts: Options) {
        self.opts = opts
        self.ring = Ring(keepMs: opts.beforeMs + Double(opts.stack) * opts.stackGapMs + 1500)
        self.dir = URL(fileURLWithPath: opts.out, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    /// The whole window's size in pixels, before the kept part is cut out —
    /// what a click's place is measured against.
    var fullW = 0, fullH = 0

    func frame(_ image: CGImage, at t: Double) {
        fullW = image.width; fullH = image.height
        guard let kept = cropped(image, opts.crop) else { return }   // refused at start; never the whole frame
        frameW = kept.width; frameH = kept.height
        ring.add(kept, at: t)
    }

    /// Writes the frame for an input and returns its file name.
    func keep(at t: Double) -> (String, Double)? {
        guard let chosen = ring.before(t, beforeMs: opts.beforeMs),
              let image = stacked(ring, at: t, beforeMs: opts.beforeMs, n: opts.stack, gapMs: opts.stackGapMs) else { return nil }
        n += 1
        let name = String(format: "f%06d.jpg", n)
        guard writeJPEG(image, to: dir.appendingPathComponent(name)) else { return nil }
        return (name, t - chosen.t)
    }

    /// Keys down now, by key code: when each went down and the frame it was
    /// kept with, so its release can say how long it was held. A tap and a
    /// hold are different inputs — in a game a tap turns, a hold walks.
    var down: [Int64: (t: Double, frame: String)] = [:]

    func key(_ name: String, code: Int64 = -1, at t: Double, frontmost: Bool) {
        guard opts.keys else { return }
        // A key typed into another app is not an input to this window.
        guard frontmost else { return }
        guard let (file, age) = keep(at: t) else { emit(["type": "dropped", "why": "no-frame-yet"]); return }
        lastInput = t
        if code >= 0 { down[code] = (t, file) }
        emit(["type": "input", "kind": "key", "key": name, "code": Int(code), "frame": file, "frame_age_ms": Int(age), "t": Int(t)])
    }

    /// A key came up: how long it was held, against the row its press made.
    func release(code: Int64, at t: Double) {
        guard let d = down.removeValue(forKey: code) else { return }
        lastInput = t
        emit(["type": "release", "code": Int(code), "frame": d.frame, "held_ms": Int(t - d.t)])
    }

    func click(global p: CGPoint, at t: Double, button: String) {
        guard opts.clicks else { return }
        guard bounds.contains(p) else { return }  // a click on another window is not an input to this one
        // Into the picture's own pixels, which is where review draws it: the
        // whole window's, less the part cut away.
        let sx = bounds.width > 0 ? Double(fullW) / bounds.width : 1
        let sy = bounds.height > 0 ? Double(fullH) / bounds.height : 1
        let kept = opts.crop?.rect(fullW, fullH) ?? CGRect(x: 0, y: 0, width: fullW, height: fullH)
        let fx = (p.x - bounds.minX) * sx, fy = (p.y - bounds.minY) * sy
        // A click outside the kept part is on something the Pigeon never sees.
        guard kept.contains(CGPoint(x: fx, y: fy)) else { emit(["type": "dropped", "why": "outside-the-kept-part"]); return }
        guard let (file, age) = keep(at: t) else { emit(["type": "dropped", "why": "no-frame-yet"]); return }
        lastInput = t
        let x = Int((fx - kept.minX).rounded()), y = Int((fy - kept.minY).rounded())
        emit(["type": "input", "kind": "click", "button": button, "x": x, "y": y,
              "frame": file, "frame_age_ms": Int(age), "t": Int(t)])
    }

    /// Nothing pressed for a while: the person is reading or waiting, and
    /// that is an answer too (NONE). One sample per quiet stretch.
    func idleTick() {
        let t = nowMs()
        guard opts.idleS > 0, t - lastInput >= opts.idleS * 1000, ring.latest != nil else { return }
        guard let (file, _) = keep(at: t + opts.beforeMs) else { return }
        lastInput = t
        emit(["type": "idle", "frame": file, "t": Int(t)])
    }
}

// MARK: - permission

func permissionLine() -> [String: Any] {
    ["type": "permission", "screen": CGPreflightScreenCaptureAccess(), "input": CGPreflightListenEventAccess()]
}

// MARK: - windows

func shareable() -> (SCShareableContent?, Error?) {
    let sem = DispatchSemaphore(value: 0)
    var content: SCShareableContent?, err: Error?
    SCShareableContent.getExcludingDesktopWindows(true, onScreenWindowsOnly: true) { c, e in content = c; err = e; sem.signal() }
    sem.wait()
    return (content, err)
}

/// The app has already asked macOS (recorder.rs); this simply tries. A refusal
/// comes back from ScreenCaptureKit itself, and is reported as one.
func listWindows() {
    let (content, err) = shareable()
    guard let content else {
        emit(["type": "error", "code": err == nil ? "no-windows" : "no-screen-permission", "message": err?.localizedDescription ?? ""]); return
    }
    let mine = ProcessInfo.processInfo.processIdentifier
    let parent = getppid()
    var list: [[String: Any]] = []
    for w in content.windows where w.windowLayer == 0 && w.frame.width >= 64 && w.frame.height >= 64 {
        guard let app = w.owningApplication else { continue }
        // Flight School's own windows are never offered.
        if app.processID == mine || app.processID == parent { continue }
        list.append(["id": w.windowID, "app": app.applicationName, "title": w.title ?? "",
                     "w": Int(w.frame.width), "h": Int(w.frame.height)])
    }
    emit(["type": "windows", "windows": list])
}

// MARK: - live recording

final class Capture: NSObject, SCStreamOutput, SCStreamDelegate {
    let session: Session
    let ci = CIContext()
    init(_ s: Session) { session = s }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        if let info = (CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]])?.first,
           let raw = info[.status] as? Int, let status = SCFrameStatus(rawValue: raw), status != .complete { return }
        let img = CIImage(cvPixelBuffer: pb)
        guard let cg = ci.createCGImage(img, from: img.extent) else { return }
        session.frame(cg, at: nowMs())
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        emit(["type": "error", "code": "window-gone", "message": error.localizedDescription])
        exit(3)
    }
}

var keepAlive: [AnyObject] = []

func record(_ opts: Options) {
    let (content, err) = shareable()
    guard let window = content?.windows.first(where: { $0.windowID == opts.window }) else {
        emit(["type": "error", "code": "window-gone", "message": err?.localizedDescription ?? "that window is not on screen"]); exit(3)
    }
    let session = Session(opts)
    session.bounds = window.frame
    session.ownerPid = window.owningApplication?.processID ?? 0

    let scale = NSScreen.screens.first(where: { $0.frame.intersects(window.frame) })?.backingScaleFactor ?? 2
    let (w, h) = captureSize(window.frame.width * scale, window.frame.height * scale, maxSide: opts.maxSide, crop: opts.crop)
    let keptSize = opts.crop?.rect(Int(w), Int(h)).size ?? CGSize(width: w, height: h)
    // A crop that keeps under a pixel of this window is refused, not recorded
    // as the whole window (format §13.2, D66) — the same code Pigeon's Eye uses.
    guard keptSize.width >= 1, keptSize.height >= 1 else {
        emit(["type": "error", "code": "crop-under-a-pixel", "message": "its crop keeps under a pixel of this window (\(Int(w)) × \(Int(h)))"]); exit(2)
    }
    let config = SCStreamConfiguration()
    config.width = Int(w); config.height = Int(h)
    config.minimumFrameInterval = CMTime(value: 1, timescale: 10)
    config.showsCursor = false
    config.queueDepth = 5
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let capture = Capture(session)
    let stream = SCStream(filter: filter, configuration: config, delegate: capture)
    do {
        try stream.addStreamOutput(capture, type: .screen, sampleHandlerQueue: DispatchQueue(label: "frames"))
    } catch { emit(["type": "error", "code": "capture-failed", "message": error.localizedDescription]); exit(3) }
    let started = DispatchSemaphore(value: 0)
    var startErr: Error?
    stream.startCapture { e in startErr = e; started.signal() }
    started.wait()
    if let startErr { emit(["type": "error", "code": "capture-failed", "message": startErr.localizedDescription]); exit(3) }
    keepAlive += [stream, capture]

    // The inputs: listen-only, so nothing is ever blocked or changed.
    let mask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue) | (1 << CGEventType.leftMouseDown.rawValue) | (1 << CGEventType.rightMouseDown.rawValue)
    let box = Unmanaged.passRetained(session).toOpaque()
    let callback: CGEventTapCallBack = { _, type, event, info in
        let s = Unmanaged<Session>.fromOpaque(info!).takeUnretainedValue()
        let t = nowMs()
        // A password field anywhere on the Mac turns secure input on. While it
        // is on, no key is kept — not even in the recorded window — and the
        // change is reported once each way.
        let secure = IsSecureEventInputEnabled()
        if secure != s.secureWas {
            s.secureWas = secure
            emit(["type": "secure-input", "on": secure])
        }
        switch type {
        case .keyDown:
            if secure { break }
            if event.getIntegerValueField(.keyboardEventAutorepeat) != 0 { break }
            var len = 0
            var chars = [UniChar](repeating: 0, count: 4)
            event.keyboardGetUnicodeString(maxStringLength: 4, actualStringLength: &len, unicodeString: &chars)
            let text = String(utf16CodeUnits: chars, count: len)
            let code = event.getIntegerValueField(.keyboardEventKeycode)
            if let name = keyName(code: code, chars: text, flags: event.flags) {
                let front = NSWorkspace.shared.frontmostApplication?.processIdentifier == s.ownerPid
                s.key(name, code: code, at: t, frontmost: front)
            }
        case .keyUp:
            if secure { break }
            s.release(code: event.getIntegerValueField(.keyboardEventKeycode), at: t)
        case .leftMouseDown: s.click(global: event.location, at: t, button: "left")
        case .rightMouseDown: s.click(global: event.location, at: t, button: "right")
        case .tapDisabledByTimeout, .tapDisabledByUserInput:
            emit(["type": "note", "what": "the input listener was paused by macOS"])
        default: break
        }
        return Unmanaged.passUnretained(event)
    }
    guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly,
                                      eventsOfInterest: CGEventMask(mask), callback: callback, userInfo: box) else {
        emit(["type": "error", "code": "no-input-permission"]); exit(2)
    }
    let src = CFMachPortCreateRunLoopSource(nil, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetMain(), src, .commonModes)
    CGEvent.tapEnable(tap: tap, enable: true)

    // The window may move while recording; clicks are placed by where it is now.
    Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
        if let b = currentBounds(window.windowID) { session.bounds = b }
        session.idleTick()
    }
    emit(["type": "ready", "window": ["id": window.windowID, "app": window.owningApplication?.applicationName ?? "",
                                     "title": window.title ?? "", "w": Int(keptSize.width), "h": Int(keptSize.height)]])
    stopOnStdin { stream.stopCapture { _ in emit(["type": "stopped", "kept": session.n]); exit(0) } }
    CFRunLoopRun()
}

/// `still --window ID --out PATH [--max-side N]`: one picture of the whole
/// window, for choosing the part to keep. Prints
/// {"type":"still","path":…,"w":…,"h":…} and exits.
final class Still: NSObject, SCStreamOutput, SCStreamDelegate {
    let ci = CIContext()
    var image: CGImage?
    let got = DispatchSemaphore(value: 0)
    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard image == nil, type == .screen, let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        if let info = (CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]])?.first,
           let raw = info[.status] as? Int, let status = SCFrameStatus(rawValue: raw), status != .complete { return }
        let img = CIImage(cvPixelBuffer: pb)
        image = ci.createCGImage(img, from: img.extent)
        if image != nil { got.signal() }
    }
}

func still(_ opts: Options) {
    let (content, err) = shareable()
    guard let window = content?.windows.first(where: { $0.windowID == opts.window }) else {
        emit(["type": "error", "code": content == nil ? "no-screen-permission" : "window-gone", "message": err?.localizedDescription ?? "that window is not on screen"]); exit(3)
    }
    let scale = NSScreen.screens.first(where: { $0.frame.intersects(window.frame) })?.backingScaleFactor ?? 2
    let (w, h) = captureSize(window.frame.width * scale, window.frame.height * scale, maxSide: opts.maxSide, crop: nil)
    let config = SCStreamConfiguration()
    config.width = Int(w); config.height = Int(h)
    config.showsCursor = false
    let grab = Still()
    let stream = SCStream(filter: SCContentFilter(desktopIndependentWindow: window), configuration: config, delegate: grab)
    do { try stream.addStreamOutput(grab, type: .screen, sampleHandlerQueue: DispatchQueue(label: "still")) }
    catch { emit(["type": "error", "code": "capture-failed", "message": error.localizedDescription]); exit(3) }
    stream.startCapture { e in if let e { emit(["type": "error", "code": "capture-failed", "message": e.localizedDescription]); exit(3) } }
    guard grab.got.wait(timeout: .now() + 5) == .success, let img = grab.image else {
        emit(["type": "error", "code": "no-frame", "message": "the window sent no picture — is it on screen?"]); exit(3)
    }
    let url = URL(fileURLWithPath: opts.out)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard writeJPEG(img, to: url) else { emit(["type": "error", "code": "write-failed"]); exit(3) }
    emit(["type": "still", "path": opts.out, "w": img.width, "h": img.height])
    exit(0)
}

func currentBounds(_ id: CGWindowID) -> CGRect? {
    guard let list = CGWindowListCopyWindowInfo([.optionIncludingWindow], id) as? [[String: Any]],
          let b = list.first?[kCGWindowBounds as String] as? [String: Any],
          let rect = CGRect(dictionaryRepresentation: b as CFDictionary) else { return nil }
    return rect
}

func stopOnStdin(_ stop: @escaping () -> Void) {
    FileHandle.standardInput.readabilityHandler = { h in
        let data = h.availableData
        if data.isEmpty || String(decoding: data, as: UTF8.self).contains("stop") {
            h.readabilityHandler = nil
            DispatchQueue.main.async(execute: stop)
        }
    }
}

// MARK: - drawn frames, for selftest and synthetic

/// A frame that says which one it is: a dark background with a bright bar
/// whose position encodes `i`, so a test can read back which frame was kept.
func drawnFrame(_ i: Int, w: Int = 160, h: Int = 144) -> CGImage {
    let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.setFillColor(CGColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(CGRect(x: (i * 8) % (w - 8), y: 0, width: 8, height: h))
    return ctx.makeImage()!
}

/// One pixel's brightness, 0–255, from the top left — for selftest.
func brightness(_ img: CGImage, x: Int, y: Int) -> Int {
    var px = [UInt8](repeating: 0, count: 4)
    guard let ctx = CGContext(data: &px, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                              space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return -1 }
    ctx.draw(img, in: CGRect(x: -x, y: y - img.height + 1, width: img.width, height: img.height))
    return Int(px[0])
}

func selftest(_ dir: String) {
    var ok: [String: Bool] = [:]
    let ring = Ring(keepMs: 2000)
    for i in 0..<10 { ring.add(drawnFrame(i), at: Double(i) * 100) }  // frames at 0,100,…,900 ms
    // An input at 850 ms keeps the frame from 80 ms earlier: 770 → the one at 700.
    ok["before_picks_the_frame_before"] = ring.before(850, beforeMs: 80)?.t == 700
    // A quiet window: the newest frame is what was on screen.
    ok["quiet_window_keeps_the_newest"] = ring.before(5000, beforeMs: 80)?.t == 900
    // An input in the first instant gets the oldest frame there is.
    ok["first_instant_gets_the_oldest"] = ring.before(10, beforeMs: 80)?.t == 0
    // Three frames stacked 250 ms apart: three frame-widths plus two gutters.
    let s = stacked(ring, at: 850, beforeMs: 80, n: 3, gapMs: 250)
    ok["stack_of_three_is_three_wide"] = s?.width == 160 * 3 + 8 && s?.height == 144
    ok["stack_of_one_is_the_frame"] = stacked(ring, at: 850, beforeMs: 80, n: 1, gapMs: 250)?.width == 160
    let url = URL(fileURLWithPath: dir).appendingPathComponent("stack3.jpg")
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    ok["jpeg_written"] = s.map { writeJPEG($0, to: url) } ?? false
    // The kept part is measured from the top left: the left half of a frame
    // whose bar is at the left keeps the bar.
    let bar = drawnFrame(0)
    let left = cropped(bar, Crop("0,0,0.5,1"))!
    ok["crop_is_the_part_asked_for"] = left.width == 80 && left.height == 144 && Crop("0.9,0,0.5,1")!.rect(160, 144).width == 16
    ok["crop_keeps_the_left"] = brightness(left, x: 2, y: 2) > 200 && brightness(cropped(bar, Crop("0.5,0,0.5,1"))!, x: 2, y: 2) < 60
    // A crop whose kept part rounds to under a pixel gives no frame at all —
    // never the whole frame (format §13.2, D66). 2% of a 20-pixel frame is 0.4.
    let small = drawnFrame(0, w: 20, h: 20)
    ok["crop_under_a_pixel_gives_no_frame"] = Crop("0,0,0.02,1") != nil && cropped(small, Crop("0,0,0.02,1")) == nil
        && Crop("0,0,0.02,1")!.rect(20, 20).width < 1 && cropped(small, Crop("0,0,0.05,1"))?.width == 1
    // A frame lit only in its top half: "0,0,1,0.5" is the top half.
    let tctx = CGContext(data: nil, width: 40, height: 40, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                         bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    tctx.setFillColor(CGColor(gray: 0, alpha: 1)); tctx.fill(CGRect(x: 0, y: 0, width: 40, height: 40))
    tctx.setFillColor(CGColor(gray: 1, alpha: 1)); tctx.fill(CGRect(x: 0, y: 20, width: 40, height: 20))  // CG's y runs up: this is the top
    let lit = tctx.makeImage()!
    ok["crop_keeps_the_top"] = brightness(lit, x: 5, y: 5) > 200 && brightness(cropped(lit, Crop("0,0,1,0.5"))!, x: 5, y: 15) > 200
        && brightness(cropped(lit, Crop("0,0.5,1,0.5"))!, x: 5, y: 5) < 60
    let (cw, ch) = captureSize(1600, 1000, maxSide: 768, crop: Crop("0,0,0.25,0.5"))
    ok["capture_size_fits_the_kept_part"] = cw == 1600 && ch == 1000 && captureSize(3200, 2000, maxSide: 768, crop: Crop("0,0,0.5,0.5")).0 == 1536
    ok["key_names"] = keyName(code: 6, chars: "z", flags: []) == "Z"
        && keyName(code: 36, chars: "\r", flags: []) == "Return"
        && keyName(code: 126, chars: "", flags: []) == "Up"
        && keyName(code: 1, chars: "s", flags: .maskCommand) == "Cmd+S"
        && keyName(code: 56, chars: "", flags: .maskShift) == nil
    emit(["type": "selftest", "ok": ok.values.allSatisfy { $0 }, "checks": ok])
}

/// `record` without the screen: drawn frames every 100 ms and a scripted run
/// of inputs, through the same Session — the same frame choice, the same
/// files and the same lines Rust reads from a live recording.
func synthetic(_ opts: Options) {
    let session = Session(opts)
    session.bounds = CGRect(x: 100, y: 100, width: 160, height: 144)
    emit(["type": "ready", "window": ["id": 0, "app": "Synthetic", "title": "drawn frames", "w": 160, "h": 144]])
    var i = 0
    let script: [(Int, String)] = [(3, "key:Z"), (5, "key:W"), (7, "click:120,130"), (9, "key:W"), (11, "click:122,128"),
                                   (13, "key:Cmd+S"), (15, "click:300,120"), (17, "click:190,200"), (19, "key:Z")]
    let t0 = nowMs()
    Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
        session.frame(drawnFrame(i), at: nowMs())
        i += 1
        if let step = script.first(where: { $0.0 == i }) {
            let t = nowMs()
            let parts = step.1.split(separator: ":")
            if parts[0] == "key" {
                session.key(String(parts[1]), code: Int64(i), at: t, frontmost: true)
                session.release(code: Int64(i), at: t + 150)
            }
            else {
                let xy = parts[1].split(separator: ",").compactMap { Double($0) }
                session.click(global: CGPoint(x: xy[0], y: xy[1]), at: t, button: "left")
            }
        }
        session.idleTick()
        if nowMs() - t0 > 3300 { timer.invalidate(); emit(["type": "stopped", "kept": session.n]); exit(0) }
    }
    stopOnStdin { emit(["type": "stopped", "kept": session.n]); exit(0) }
    CFRunLoopRun()
}

// MARK: - fingerprints, for finding rows that look the same

/// A 64-bit difference hash: the picture shrunk to 9×8 in grey, one bit per
/// neighbouring pair (is the left brighter?). Two frames of the same screen
/// differ in a bit or two; different screens in dozens. What Flight School
/// uses to find rows whose pictures match but whose answers do not.
func dhash(_ url: URL) -> UInt64? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let w = 9, h = 8
    var px = [UInt8](repeating: 0, count: w * h)
    guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                              space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
    ctx.interpolationQuality = .medium
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    var bits: UInt64 = 0
    for y in 0..<h { for x in 0..<(w - 1) { bits = (bits << 1) | (px[y * w + x] > px[y * w + x + 1] ? 1 : 0) } }
    return bits
}

/// The picture itself, small: 32 × 32 grey levels, row by row. Two frames of
/// one screen match here to within compression noise; a sprite that moved, a
/// number that changed or a menu that opened changes some cell by far more.
/// This is what "the same picture" is judged on — the 64-bit hash only finds
/// the candidates.
func signature(_ url: URL) -> [UInt8]? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let n = 32
    var px = [UInt8](repeating: 0, count: n * n)
    guard let ctx = CGContext(data: &px, width: n, height: n, bitsPerComponent: 8, bytesPerRow: n,
                              space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: n, height: n))
    return px
}

/// `hash` reads file paths, one per line, on stdin and prints one line:
/// {"type":"hashes","hashes":{"<path>":"<16 hex digits>"},"sigs":{"<path>":"<base64 of 1024 bytes>"}}.
/// A file it cannot read is left out of both.
func hashAll() {
    let text = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
    var out: [String: String] = [:]
    var sigs: [String: String] = [:]
    for line in text.split(separator: "\n") {
        let path = String(line)
        let url = URL(fileURLWithPath: path)
        if let h = dhash(url) { out[path] = String(format: "%016llx", h) }
        if let s = signature(url) { sigs[path] = Data(s).base64EncodedString() }
    }
    emit(["type": "hashes", "hashes": out, "sigs": sigs])
}

/// `resize --max-side N [--crop x,y,w,h]` reads `<from>\t<to>` lines on stdin
/// and writes each picture again in the recorder's own order: the whole
/// picture shrunk so that the part the crop keeps has its longer side at most
/// N pixels (never enlarged), then cut to the crop (`captureSize`, then
/// `cropped`, exactly as `record` does) — as JPEG, or PNG when `<to>` ends in
/// .png. It prints one line:
/// {"type":"resized","done":<count>,"failed":["<from>", …]}.
/// Teach uses it to train at a smaller size than a table was recorded at; the
/// same N goes into the Pigeon, whose eye then shrinks what it sees to match.
func resizeAll(_ maxSide: Int, crop: Crop? = nil) {
    let text = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
    var done = 0
    var failed: [String] = []
    for line in text.split(separator: "\n") {
        let parts = line.split(separator: "\t", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { continue }
        let (from, to) = (URL(fileURLWithPath: parts[0]), URL(fileURLWithPath: parts[1]))
        guard let src = CGImageSourceCreateWithURL(from as CFURL, nil),
              let whole = CGImageSourceCreateImageAtIndex(src, 0, nil) else { failed.append(parts[0]); continue }
        let (sw, sh) = captureSize(Double(whole.width), Double(whole.height), maxSide: maxSide, crop: crop)
        var scaled = whole
        if Int(sw) != whole.width || Int(sh) != whole.height {
            let w = max(1, Int(sw)), h = max(1, Int(sh))
            guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { failed.append(parts[0]); continue }
            ctx.interpolationQuality = .high
            ctx.draw(whole, in: CGRect(x: 0, y: 0, width: w, height: h))
            guard let small = ctx.makeImage() else { failed.append(parts[0]); continue }
            scaled = small
        }
        guard let out = cropped(scaled, crop) else { failed.append(parts[0]); continue }   // under a pixel: never the whole picture
        try? FileManager.default.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
        let png = to.pathExtension.lowercased() == "png"
        let ok: Bool
        if png, let dest = CGImageDestinationCreateWithURL(to as CFURL, UTType.png.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(dest, out, nil)
            ok = CGImageDestinationFinalize(dest)
        } else {
            ok = writeJPEG(out, to: to)
        }
        if ok { done += 1 } else { failed.append(parts[0]) }
    }
    emit(["type": "resized", "done": done, "failed": failed])
}

// MARK: - main

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "" {
case "permission": emit(permissionLine())
case "request":
    _ = CGRequestScreenCaptureAccess()
    _ = CGRequestListenEventAccess()
    emit(permissionLine())
case "windows": listWindows()
case "record": record(Options(args.dropFirst(2)))
case "still": still(Options(args.dropFirst(2)))
case "synthetic": synthetic(Options(args.dropFirst(2)))
case "selftest": selftest(args.count > 2 ? args[2] : NSTemporaryDirectory())
case "hash": hashAll()
case "resize":
    let i = args.firstIndex(of: "--max-side"), c = args.firstIndex(of: "--crop")
    resizeAll(max(64, i.flatMap { args.count > $0 + 1 ? Int(args[$0 + 1]) : nil } ?? 768),
              crop: c.flatMap { args.count > $0 + 1 ? Crop(args[$0 + 1]) : nil })
default:
    emit(["type": "error", "code": "usage", "message": "permission | request | windows | record | still | synthetic | selftest | hash | resize"])
    exit(64)
}
