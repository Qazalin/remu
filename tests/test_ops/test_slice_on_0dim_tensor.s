
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001500 <E_n12>:
	s_load_b64 s[0:1], s[0:1], null                            // 000000001500: F4040000 F8000000
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0x3e15f480      // 000000001508: CA100080 000000FF 3E15F480
	s_waitcnt lgkmcnt(0)                                       // 000000001514: BF89FC07
	global_store_b32 v0, v1, s[0:1]                            // 000000001518: DC6A0000 00000100
	s_nop 0                                                    // 000000001520: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001524: BFB60003
	s_endpgm                                                   // 000000001528: BFB00000
