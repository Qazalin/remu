
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_2_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v1, 0      // 000000001608: CA1000FF 00000080 FF800000
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_clause 0x1                                               // 000000001618: BF850001
	s_load_b64 s[4:5], s[2:3], null                            // 00000000161C: F4040101 F8000000
	s_load_b64 s[2:3], s[2:3], 0xc                             // 000000001624: F4040081 F800000C
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	v_max3_f32 v0, s5, s4, v0                                  // 000000001630: D61C0000 04000805
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001638: BF870001
	v_max3_f32 v0, s3, s2, v0                                  // 00000000163C: D61C0000 04000403
	global_store_b32 v1, v0, s[0:1]                            // 000000001644: DC6A0000 00000001
	s_nop 0                                                    // 00000000164C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001650: BFB60003
	s_endpgm                                                   // 000000001654: BFB00000
