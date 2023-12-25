
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n40>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001614: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001618: 86039F0F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000161C: BF8704D9
	s_lshl_b64 s[8:9], s[2:3], 1                               // 000000001620: 84888102
	s_waitcnt lgkmcnt(0)                                       // 000000001624: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001628: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000162C: 82070907
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001630: 84828202
	s_add_u32 s0, s0, s2                                       // 000000001634: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001638: 82010301
	s_load_b32 s0, s[0:1], null                                // 00000000163C: F4000000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001644: 7E000280
	global_load_i16 v1, v0, s[6:7]                             // 000000001648: DC4E0000 01060000
	s_waitcnt vmcnt(0)                                         // 000000001650: BF8903F7
	v_cvt_f32_i32_e32 v1, v1                                   // 000000001654: 7E020B01
	s_waitcnt lgkmcnt(0)                                       // 000000001658: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000165C: BF870001
	v_add_f32_e32 v1, s0, v1                                   // 000000001660: 06020200
	s_add_u32 s0, s4, s2                                       // 000000001664: 80000204
	s_addc_u32 s1, s5, s3                                      // 000000001668: 82010305
	global_store_b32 v0, v1, s[0:1]                            // 00000000166C: DC6A0000 00000100
	s_nop 0                                                    // 000000001674: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001678: BFB60003
	s_endpgm                                                   // 00000000167C: BFB00000
