
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_16384>:
	s_load_b128 s[16:19], s[0:1], null                         // 000000001600: F4080400 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_mov_b64 s[20:21], 0                                      // 00000000160C: BE940180
	s_nop 0                                                    // 000000001610: BF800000
	s_nop 0                                                    // 000000001614: BF800000
	s_nop 0                                                    // 000000001618: BF800000
	s_nop 0                                                    // 00000000161C: BF800000
	s_nop 0                                                    // 000000001620: BF800000
	s_nop 0                                                    // 000000001624: BF800000
	s_nop 0                                                    // 000000001628: BF800000
	s_nop 0                                                    // 00000000162C: BF800000
	s_nop 0                                                    // 000000001630: BF800000
	s_nop 0                                                    // 000000001634: BF800000
	s_nop 0                                                    // 000000001638: BF800000
	s_nop 0                                                    // 00000000163C: BF800000
	s_waitcnt lgkmcnt(0)                                       // 000000001640: BF89FC07
	s_add_u32 s0, s18, s20                                     // 000000001644: 80001412
	s_addc_u32 s1, s19, s21                                    // 000000001648: 82011513
	s_add_u32 s20, s20, 64                                     // 00000000164C: 8014C014
	s_load_b512 s[0:15], s[0:1], null                          // 000000001650: F4100000 F8000000
	s_addc_u32 s21, s21, 0                                     // 000000001658: 82158015
	s_cmp_eq_u32 s20, 0x10000                                  // 00000000165C: BF06FF14 00010000
	s_waitcnt lgkmcnt(0)                                       // 000000001664: BF89FC07
	v_add_f32_e32 v0, s0, v0                                   // 000000001668: 06000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000166C: BF870091
	v_add_f32_e32 v0, s1, v0                                   // 000000001670: 06000001
	v_add_f32_e32 v0, s2, v0                                   // 000000001674: 06000002
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001678: BF870091
	v_add_f32_e32 v0, s3, v0                                   // 00000000167C: 06000003
	v_add_f32_e32 v0, s4, v0                                   // 000000001680: 06000004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001684: BF870091
	v_add_f32_e32 v0, s5, v0                                   // 000000001688: 06000005
	v_add_f32_e32 v0, s6, v0                                   // 00000000168C: 06000006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001690: BF870091
	v_add_f32_e32 v0, s7, v0                                   // 000000001694: 06000007
	v_add_f32_e32 v0, s8, v0                                   // 000000001698: 06000008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000169C: BF870091
	v_add_f32_e32 v0, s9, v0                                   // 0000000016A0: 06000009
	v_add_f32_e32 v0, s10, v0                                  // 0000000016A4: 0600000A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A8: BF870091
	v_add_f32_e32 v0, s11, v0                                  // 0000000016AC: 0600000B
	v_add_f32_e32 v0, s12, v0                                  // 0000000016B0: 0600000C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016B4: BF870091
	v_add_f32_e32 v0, s13, v0                                  // 0000000016B8: 0600000D
	v_add_f32_e32 v0, s14, v0                                  // 0000000016BC: 0600000E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016C0: BF870001
	v_add_f32_e32 v0, s15, v0                                  // 0000000016C4: 0600000F
	s_cbranch_scc0 65501                                       // 0000000016C8: BFA1FFDD <r_16384+0x40>
	v_mov_b32_e32 v1, 0                                        // 0000000016CC: 7E020280
	global_store_b32 v1, v0, s[16:17]                          // 0000000016D0: DC6A0000 00100001
	s_nop 0                                                    // 0000000016D8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016DC: BFB60003
	s_endpgm                                                   // 0000000016E0: BFB00000
