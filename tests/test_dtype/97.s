
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n30>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001614: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001618: 86039F0F
	v_mov_b32_e32 v1, 0                                        // 00000000161C: 7E020280
	s_lshl_b64 s[8:9], s[2:3], 2                               // 000000001620: 84888202
	s_waitcnt lgkmcnt(0)                                       // 000000001624: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001628: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000162C: 82070907
	s_lshl_b64 s[2:3], s[2:3], 3                               // 000000001630: 84828302
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001634: BF870009
	s_add_u32 s0, s0, s2                                       // 000000001638: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000163C: 82010301
	s_load_b64 s[0:1], s[0:1], null                            // 000000001640: F4040000 F8000000
	s_load_b32 s2, s[6:7], null                                // 000000001648: F4000083 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001650: BF89FC07
	s_clz_i32_u32 s3, s1                                       // 000000001654: BE830A01
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001658: BF870499
	s_min_u32 s3, s3, 32                                       // 00000000165C: 8983A003
	s_lshl_b64 s[0:1], s[0:1], s3                              // 000000001660: 84800300
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001664: BF870499
	s_min_u32 s0, s0, 1                                        // 000000001668: 89808100
	s_or_b32 s0, s1, s0                                        // 00000000166C: 8C000001
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001670: BF870009
	v_cvt_f32_u32_e32 v0, s0                                   // 000000001674: 7E000C00
	s_sub_i32 s0, 32, s3                                       // 000000001678: 818003A0
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 00000000167C: BF870481
	v_ldexp_f32 v0, v0, s0                                     // 000000001680: D71C0000 00000100
	s_add_u32 s0, s4, s8                                       // 000000001688: 80000804
	s_addc_u32 s1, s5, s9                                      // 00000000168C: 82010905
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001690: BF870001
	v_add_f32_e32 v0, s2, v0                                   // 000000001694: 06000002
	global_store_b32 v1, v0, s[0:1]                            // 000000001698: DC6A0000 00000001
	s_nop 0                                                    // 0000000016A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016A4: BFB60003
	s_endpgm                                                   // 0000000016A8: BFB00000
