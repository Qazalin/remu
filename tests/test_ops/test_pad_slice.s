
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_n6>:
	s_load_b64 s[0:1], s[0:1], null                            // 000000001600: F4040000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_waitcnt lgkmcnt(0)                                       // 00000000160C: BF89FC07
	global_store_b32 v0, v0, s[0:1]                            // 000000001610: DC6A0000 00000000
	s_nop 0                                                    // 000000001618: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000161C: BFB60003
	s_endpgm                                                   // 000000001620: BFB00000
