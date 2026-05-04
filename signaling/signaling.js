/**
 * WebRTC 시그널링 서버
 *
 * 역할: SDP offer/answer, ICE candidate 메시지만 중계.
 *       실제 activation tensor는 P2P DataChannel로 직접 전송됨 (서버 무관).
 *
 * 프로토콜:
 *   join          { type, room, nodeId }
 *   offer         { type, room, targetId, sdp }
 *   answer        { type, room, targetId, sdp }
 *   ice_candidate { type, room, targetId, candidate }
 *
 * 실행: node signaling.js [--port 8765]
 */

const WebSocket = require("ws");
const args = process.argv.slice(2);
const PORT = args.includes("--port")
  ? parseInt(args[args.indexOf("--port") + 1])
  : 8765;

const server = new WebSocket.Server({ port: PORT });

// room_id → Set<WebSocket>
const rooms = new Map();

server.on("connection", (ws) => {
  ws.isAlive = true;
  ws.on("pong", () => { ws.isAlive = true; });

  ws.on("message", (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      ws.send(JSON.stringify({ type: "error", message: "invalid JSON" }));
      return;
    }

    switch (msg.type) {
      case "join": {
        if (!msg.room || !msg.nodeId) break;
        if (!rooms.has(msg.room)) rooms.set(msg.room, new Set());

        const room = rooms.get(msg.room);
        room.add(ws);
        ws.room = msg.room;
        ws.nodeId = msg.nodeId;

        // 방에 있는 기존 노드들에게 신규 참여자 알림
        broadcast(msg.room, ws, { type: "peer_joined", nodeId: msg.nodeId });

        // 신규 참여자에게 기존 노드 목록 전달
        const peers = [...room]
          .filter((p) => p !== ws)
          .map((p) => p.nodeId);
        ws.send(JSON.stringify({ type: "room_state", peers }));

        console.log(`[${msg.room}] ${msg.nodeId} joined (${room.size} nodes)`);
        break;
      }

      case "offer":
      case "answer":
      case "ice_candidate": {
        // targetId 지정 시 단일 전달, 없으면 broadcast
        const envelope = { ...msg, from: ws.nodeId };
        if (msg.targetId) {
          forward(ws.room, msg.targetId, envelope);
        } else {
          broadcast(ws.room, ws, envelope);
        }
        break;
      }

      default:
        ws.send(JSON.stringify({ type: "error", message: `unknown type: ${msg.type}` }));
    }
  });

  ws.on("close", () => {
    if (!ws.room) return;
    rooms.get(ws.room)?.delete(ws);
    broadcast(ws.room, ws, { type: "peer_left", nodeId: ws.nodeId });
    console.log(`[${ws.room}] ${ws.nodeId} left`);
  });

  ws.on("error", (err) => console.error(`[ws error] ${err.message}`));
});

// 빈 방 정리 주기: 60초
setInterval(() => {
  for (const [id, room] of rooms) {
    if (room.size === 0) rooms.delete(id);
  }
}, 60_000);

// Heartbeat: 끊긴 연결 감지 (30초)
setInterval(() => {
  server.clients.forEach((ws) => {
    if (!ws.isAlive) { ws.terminate(); return; }
    ws.isAlive = false;
    ws.ping();
  });
}, 30_000);

function broadcast(room, sender, msg) {
  const str = JSON.stringify(msg);
  for (const peer of rooms.get(room) ?? []) {
    if (peer !== sender && peer.readyState === WebSocket.OPEN) {
      peer.send(str);
    }
  }
}

function forward(room, targetId, msg) {
  const str = JSON.stringify(msg);
  for (const peer of rooms.get(room) ?? []) {
    if (peer.nodeId === targetId && peer.readyState === WebSocket.OPEN) {
      peer.send(str);
      return;
    }
  }
}

console.log(`Signaling server running on ws://0.0.0.0:${PORT}`);