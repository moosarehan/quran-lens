import React, { useEffect, useRef, useState } from 'react';
import { BookOpen, Menu, X, Plus, Search, Trash2, Pencil, Sun, Moon, Info, ArrowUp, ArrowUpRight, Copy, MessageSquare, ShieldCheck, Leaf, Star, Heart } from './Icons';
import { request, streamAnswer } from './api';

const suggestions = [
  ['Thematic search', 'What does the Quran say about patience?', Leaf],
  ['Specific ayah', 'Show Ayat al-Kursi — 2:255', Star],
  ['Guidance & ethics', 'Find verses about kindness to parents', Heart],
  ['Chapter overview', 'Show Surah Al-Ikhlas', BookOpen],
];

function Logo() { return <div className="brand"><span className="logo"><BookOpen size={25}/></span><div><strong>QuranLens</strong><small>Explore the Quran through its verses</small></div></div>; }
function VerseCard({ verse, onOpen, onCopy }) {
  return <article className="verse-card"><header><div><BookOpen size={17}/><strong>{verse.surah_name || 'Quran verse'}</strong></div><button className="reference" onClick={() => onOpen(verse.verse_key)}>{verse.verse_key}</button></header>
    <p className="arabic" dir="rtl" lang="ar">{verse.arabic_text}</p><div className="translation-label">TRANSLATION · SAHEEH INTERNATIONAL</div><p className="translation">{verse.english_translation}</p>
    <footer><button onClick={() => onCopy(`${verse.arabic_text}\n${verse.english_translation} [${verse.verse_key}]`)}><Copy size={14}/> Copy verse</button><button onClick={() => onOpen(verse.verse_key)}>Inspect context <ArrowUpRight size={15}/></button></footer></article>;
}
function Answer({ text, onOpen }) {
  return <div className="answer-text">{text.split(/(\[\d{1,3}:\d{1,3}\])/g).map((part, i) => /^\[\d+:\d+\]$/.test(part) ? <button key={i} className="inline-citation" onClick={() => onOpen(part.slice(1, -1))}>{part}</button> : <React.Fragment key={i}>{part}</React.Fragment>)}</div>;
}
function History({ chats, active, busy, onSelect, onNew, onDelete, onRename, onClose }) {
  const [search, setSearch] = useState('');
  const now = new Date();
  const group = chat => { const date = new Date(chat.updated); if (date.toDateString() === now.toDateString()) return 'Today'; const yesterday = new Date(now); yesterday.setDate(now.getDate()-1); return date.toDateString() === yesterday.toDateString() ? 'Yesterday' : 'Earlier'; };
  const filtered = chats.filter(c => c.title.toLowerCase().includes(search.toLowerCase()));
  return <><div className="sidebar-top"><Logo/><button className="icon-button mobile" aria-label="Close history" onClick={onClose}><X/></button></div><button className="new-chat" disabled={busy} onClick={onNew}><Plus size={18}/> New conversation</button><label className="search"><Search size={16}/><input placeholder="Search conversations" aria-label="Search conversations" value={search} onChange={e=>setSearch(e.target.value)}/></label><div className="history-list">
    {['Today','Yesterday','Earlier'].map(label => { const items = filtered.filter(c => group(c) === label); return items.length ? <section key={label}><h3>{label} <span>{items.length}</span></h3>{items.map(chat => <div className={`history-item ${active === chat.id ? 'selected' : ''}`} key={chat.id}><button disabled={busy} onClick={()=>onSelect(chat.id)}><MessageSquare size={15}/><span>{chat.title}</span></button><div className="history-actions"><button disabled={busy} aria-label={`Rename ${chat.title}`} onClick={()=>onRename(chat)}><Pencil size={13}/></button><button disabled={busy} aria-label={`Delete ${chat.title}`} onClick={()=>onDelete(chat)}><Trash2 size={13}/></button></div></div>)}</section> : null; })}
    {!filtered.length && <p className="history-empty">{search ? 'No matching conversations.' : 'Your conversations will appear here.'}</p>}
  </div><div className="sidebar-bottom"><ShieldCheck size={16}/><div>Grounded in Quran verses<small>Tanzil · Saheeh International</small></div></div></>;
}

export default function App() {
  const [chats,setChats]=useState([]), [active,setActive]=useState(null), [messages,setMessages]=useState([]), [input,setInput]=useState(''), [busy,setBusy]=useState(false), [status,setStatus]=useState(''), [error,setError]=useState(''), [drawer,setDrawer]=useState(false), [dialog,setDialog]=useState(null), [rename,setRename]=useState(''), [citation,setCitation]=useState(null), [toast,setToast]=useState('');
  const [dark,setDark]=useState(()=>localStorage.getItem('quranlens-theme')==='dark');
  const bottom=useRef(null), textbox=useRef(null), selection=useRef(0);
  const refresh = async () => setChats(await request('/conversations'));
  useEffect(()=>{ refresh().catch(e=>setError(e.message)); const saved=localStorage.getItem('quranlens-thread'); if(saved) openChat(saved); },[]);
  useEffect(()=>{ document.documentElement.dataset.theme=dark?'dark':'light'; localStorage.setItem('quranlens-theme',dark?'dark':'light'); },[dark]);
  useEffect(()=>{ if(toast){const timer=setTimeout(()=>setToast(''),2500);return()=>clearTimeout(timer);} },[toast]);
  useEffect(()=>{bottom.current?.scrollIntoView({behavior:'smooth'});},[messages,status]);
  async function openChat(id) {
    const version=++selection.current; setError('');
    try {const chat=await request(`/conversations/${id}`); if(version!==selection.current)return;setActive(id);setMessages(chat.messages);localStorage.setItem('quranlens-thread',id);setDrawer(false);} catch(e){if(version===selection.current)setError(e.message);}
  }
  async function newChat() {try {++selection.current;const chat=await request('/conversations',{method:'POST'});setActive(chat.id);setMessages([]);setInput('');setError('');setDrawer(false);localStorage.setItem('quranlens-thread',chat.id);await refresh();textbox.current?.focus();}catch(e){setError(e.message);}}
  async function send(event) {
    event?.preventDefault();const question=input.trim();if(!question||busy)return;
    setBusy(true);setError('');setStatus('Finding relevant Quran verses…');setInput('');
    let id=active;
    try {
      if(!id){const chat=await request('/conversations',{method:'POST'});id=chat.id;setActive(id);localStorage.setItem('quranlens-thread',id);}
      setMessages(previous=>[...previous,{role:'user',content:question},{role:'assistant',content:'',verses:[],pending:true}]);
      await streamAnswer(id,question,event=>{
        if(event.type==='status')setStatus(event.message);
        if(event.type==='answer_token')setMessages(previous=>previous.map((m,i)=>i===previous.length-1?{...m,content:m.content+event.text}:m));
        if(event.type==='result')setMessages(previous=>previous.map((m,i)=>i===previous.length-1?{role:'assistant',content:event.answer,...event,pending:false}:m));
      });
      const chat=await request(`/conversations/${id}`);setMessages(chat.messages);
    } catch(e){setError(e.message);setInput(question);if(id){try{const chat=await request(`/conversations/${id}`);setMessages(chat.messages);}catch{}}}
    finally{setBusy(false);setStatus('');refresh().catch(()=>{});}
  }
  async function confirmDialog() {try {if(dialog.type==='delete'){await request(`/conversations/${dialog.chat.id}`,{method:'DELETE'});if(active===dialog.chat.id){++selection.current;setActive(null);setMessages([]);localStorage.removeItem('quranlens-thread');}}else{await request(`/conversations/${dialog.chat.id}`,{method:'PATCH',body:JSON.stringify({title:rename.trim()})});}setDialog(null);await refresh();}catch(e){setError(e.message);}}
  async function openCitation(key) {setCitation({loading:true});try {const [s,a]=key.split(':').map(Number);const values=await Promise.all([a>1?request(`/verses/${s}/${a-1}`).catch(()=>null):null,request(`/verses/${s}/${a}`),request(`/verses/${s}/${a+1}`).catch(()=>null)]);setCitation({verse:values[1],neighbors:[values[0],values[2]].filter(Boolean)});}catch(e){setCitation({error:e.message});}}
  async function copy(text){try{await navigator.clipboard.writeText(text);setToast('Copied to clipboard');}catch{setToast('Clipboard is unavailable in this browser.');}}
  return <div className="app"><aside className={`sidebar ${drawer?'open':''}`}><History chats={chats} active={active} busy={busy} onSelect={openChat} onNew={newChat} onDelete={chat=>setDialog({type:'delete',chat})} onRename={chat=>{setRename(chat.title);setDialog({type:'rename',chat});}} onClose={()=>setDrawer(false)}/></aside>{drawer&&<button className="drawer-backdrop" aria-label="Close history" onClick={()=>setDrawer(false)}/>}
    <div className="workspace"><header className="navbar"><div className="nav-left"><button className="icon-button mobile" aria-label="Open conversation history" onClick={()=>setDrawer(true)}><Menu/></button><span className="desktop-label">Explore the Quran</span><div className="mobile"><Logo/></div></div><div className="nav-actions"><button className="icon-button" aria-label="Toggle theme" onClick={()=>setDark(!dark)}>{dark?<Sun size={20}/>:<Moon size={20}/>}</button><button className="about-button" onClick={()=>setDialog({type:'about'})}><Info size={18}/><span>About</span></button></div></header>
    <main>{messages.length===0?<section className="welcome"><div className="hero-emblem"><BookOpen size={49}/><span>✦</span></div><div className="eyebrow">A SPACE FOR UNDERSTANDING</div><h1>What would you like<br/>to explore?</h1><p>Find verses, explore themes, and read the Quran<br className="desktop"/> in Arabic with translation.</p><div className="bismillah" dir="rtl" lang="ar">بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ</div><div className="suggestion-heading">Guided inquiries <span>A starting point for your study</span></div><div className="suggestions">{suggestions.map(([label,text,Icon])=><button disabled={busy} key={label} onClick={()=>{setInput(text);textbox.current?.focus();}}><div><Icon size={18}/><span>{label}</span><ArrowUpRight size={17}/></div><strong>{text}</strong></button>)}</div></section>:<section className="conversation"><div className="conversation-meta"><span><i/> {chats.find(c=>c.id===active)?.title || 'Quran study'}</span><small>Quran-only conversation</small></div>{messages.map((message,i)=><div className={`message ${message.role}`} key={message.id||`pending-${i}`}>{message.role==='user'?<div className="user-bubble">{message.content}</div>:<><div className="assistant-label"><span><BookOpen size={16}/></span> QuranLens <small><ShieldCheck size={13}/> Quran sources</small></div><Answer text={message.content} onOpen={openCitation}/>{message.verses?.length>0&&<div className="verse-list">{message.verses.map(v=><VerseCard key={v.verse_key} verse={v} onOpen={openCitation} onCopy={copy}/>)}</div>}{message.notices?.map((n,j)=><p className="notice" key={j}>{n}</p>)}{message.content&&!message.pending&&<button className="copy-response" onClick={()=>copy(message.content)}><Copy size={14}/> Copy response</button>}</>}</div>)}<div ref={bottom}/></section>}
    </main><div className="composer-area">{busy&&<div className="status" role="status"><span className="pulse"/>{status||'Preparing your answer…'}</div>}{error&&<div className="error" role="alert">{error}<button aria-label="Dismiss error" onClick={()=>setError('')}><X size={15}/></button></div>}<form className="composer" onSubmit={send}><textarea ref={textbox} aria-label="Ask a Quran question" placeholder="Ask a question or enter a verse reference…" maxLength={4000} value={input} rows={2} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();send();}}}/><button className="send" disabled={busy||!input.trim()} aria-label="Send question"><ArrowUp size={21}/></button></form><div className="composer-note"><ShieldCheck size={12}/> Answers are grounded in Quran verses. Explore citations for context.<span>Enter to send</span></div></div></div>
    {citation&&<div className="overlay" onClick={()=>setCitation(null)}><section className="citation-panel" role="dialog" aria-modal="true" aria-label="Ayah citation context" onClick={e=>e.stopPropagation()}><header><div><BookOpen size={19}/><strong>Ayah citation context</strong></div><button className="icon-button" aria-label="Close citation" onClick={()=>setCitation(null)}><X/></button></header>{citation.loading?<p>Loading Quran context…</p>:citation.error?<p role="alert">{citation.error}</p>:<><VerseCard verse={citation.verse} onCopy={copy} onOpen={()=>{}}/><h3>Neighboring verses in flow</h3>{citation.neighbors.map(v=><VerseCard key={v.verse_key} verse={v} onCopy={copy} onOpen={openCitation}/>)}</>}</section></div>}
    {dialog&&<div className="overlay" onClick={()=>setDialog(null)}><section className="modal" role="dialog" aria-modal="true" aria-label={dialog.type==='about'?'About QuranLens':`${dialog.type} conversation`} onClick={e=>e.stopPropagation()}>{dialog.type==='about'?<><Logo/><h2>A space for Quran study</h2><p>Read canonical Arabic verses and the Saheeh International translation. Responses are grounded in retrieved Quran text; citations let you explore the evidence and surrounding verses.</p><button className="primary-button" onClick={()=>setDialog(null)}>Continue exploring</button></>:<><h2>{dialog.type==='delete'?'Delete conversation?':'Rename conversation'}</h2>{dialog.type==='delete'?<p>This removes “{dialog.chat.title}” and all its messages permanently.</p>:<input autoFocus aria-label="Conversation title" value={rename} maxLength={100} onChange={e=>setRename(e.target.value)}/>}<div className="modal-actions"><button onClick={()=>setDialog(null)}>Cancel</button><button disabled={dialog.type==='rename'&&!rename.trim()} className={dialog.type==='delete'?'danger':'primary-button'} onClick={confirmDialog}>{dialog.type==='delete'?'Delete':'Save title'}</button></div></>}</section></div>}{toast&&<div className="toast" role="status">{toast}</div>}
  </div>;
}
